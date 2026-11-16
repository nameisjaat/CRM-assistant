"""Decide whether a pipeline result is the same answer as a golden result.

Shared by the evals (eval_generate.py now, the whole-pipeline eval later), so every eval
judges "correct" the same way. Plain code: no LLM, no database.

    match, reason = results_match(golden["expected_result"], columns, rows,
                                  golden["expected_shape"]["order_matters"])

Rules (agreed with the project owner):
  - Column names and column order are ignored. Every golden column must match some
    pipeline column by its values; extra pipeline columns are fine.
  - Row order is ignored unless order_matters is true (then rows compare by position).
  - The row count must be equal (a missing LIMIT is a wrong answer).
  - Numbers match within 0.01 (an unrounded 4.0864 matches 4.09; a 0-1 ratio 0.0677
    does not match 6.77).
  - Month labels are normalised: a golden '2017-01' matches '2017-01', '2017-01-01' or
    '2017-01-01T00:00:00-02:00'. A bare month number (1) does not match.
  - An empty golden result matches 0 rows, or rows where every value is NULL.
"""
import re

TOLERANCE = 0.01
MONTH = re.compile(r"^\d{4}-\d{2}$")                  # golden month label, e.g. 2017-01


# ---------------------------------------------------------------- values

def _as_number(v) -> float | None:
    """A number, or a string holding one ('2016'); None otherwise. Booleans are not numbers."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def values_equal(expected, actual) -> bool:
    """One golden value against one pipeline value."""
    if expected is None or actual is None:
        return expected is None and actual is None
    e_num, a_num = _as_number(expected), _as_number(actual)
    if e_num is not None and a_num is not None:
        return abs(e_num - a_num) <= TOLERANCE + 1e-9
    if isinstance(expected, str) and MONTH.match(expected) and isinstance(actual, str):
        return actual[:7] == expected                 # 2017-01-01T00:00:00-02:00 -> 2017-01
    return str(expected) == str(actual)


def _sort_key(v):
    """Orders a column's values so two equal columns line up value by value."""
    if v is None:
        return (0, 0.0, "")
    n = _as_number(v)
    if n is not None:
        return (1, n, "")
    s = str(v)
    return (2, 0.0, s[:7] if re.match(r"^\d{4}-\d{2}", s) else s)


# ---------------------------------------------------------------- columns and rows

def _column(rows: list[list], i: int) -> list:
    return [row[i] for row in rows]


def _columns_equal(expected: list, actual: list, ordered: bool) -> bool:
    if not ordered:
        expected, actual = sorted(expected, key=_sort_key), sorted(actual, key=_sort_key)
    return all(values_equal(e, a) for e, a in zip(expected, actual))


def _rows_equal(expected_rows: list[list], actual_rows: list[list], ordered: bool) -> bool:
    if not ordered:
        expected_rows = sorted(expected_rows, key=lambda r: [_sort_key(v) for v in r])
        actual_rows = sorted(actual_rows, key=lambda r: [_sort_key(v) for v in r])
    return all(values_equal(e, a) for er, ar in zip(expected_rows, actual_rows)
               for e, a in zip(er, ar))


def _is_empty(rows: list[list]) -> bool:
    return not rows or all(v is None for row in rows for v in row)


def _first_difference(expected_rows, actual_rows, mapping, ordered) -> str:
    projected = [[row[k] for k in mapping] for row in actual_rows]
    if not ordered:
        expected_rows = sorted(expected_rows, key=lambda r: [_sort_key(v) for v in r])
        projected = sorted(projected, key=lambda r: [_sort_key(v) for v in r])
    for i, (er, ar) in enumerate(zip(expected_rows, projected), 1):
        if not all(values_equal(e, a) for e, a in zip(er, ar)):
            return f"row {i} differs: expected {er}, got {ar}"
    return "rows differ"


# ---------------------------------------------------------------- entry point

def results_match(expected_result: dict, actual_columns: list[str], actual_rows: list[list],
                  order_matters: bool) -> tuple[bool, str]:
    """Is the pipeline's result the same answer as the golden result? Returns (match, reason)."""
    exp_cols, exp_rows = expected_result["columns"], expected_result["rows"]

    if _is_empty(exp_rows):
        if _is_empty(actual_rows):
            return True, "both empty"
        return False, f"expected no rows, got {len(actual_rows)}"
    if len(actual_rows) != len(exp_rows):
        return False, f"row count {len(actual_rows)}, expected {len(exp_rows)}"

    # candidate pipeline columns for each golden column, matched by values alone
    candidates = []
    for j, name in enumerate(exp_cols):
        exp_col = _column(exp_rows, j)
        found = [k for k in range(len(actual_columns))
                 if _columns_equal(exp_col, _column(actual_rows, k), order_matters)]
        if not found:
            return False, f"no column matches golden column {name} (values {exp_col[:3]}...)"
        candidates.append(found)

    # pick one distinct pipeline column per golden column so that whole rows match
    first_mapping = None
    for mapping in _mappings(candidates):
        first_mapping = first_mapping or mapping
        projected = [[row[k] for k in mapping] for row in actual_rows]
        if _rows_equal(exp_rows, projected, order_matters):
            matched = ", ".join(f"{exp_cols[j]}={actual_columns[k]}" for j, k in enumerate(mapping))
            return True, f"match ({matched})"
    if first_mapping is None:
        return False, "golden columns can't each get a different pipeline column"
    return False, _first_difference(exp_rows, actual_rows, first_mapping, order_matters)


def _mappings(candidates: list[list[int]]):
    """Every way to give each golden column a different pipeline column from its candidates."""
    def walk(j: int, used: tuple):
        if j == len(candidates):
            yield used
            return
        for k in candidates[j]:
            if k not in used:
                yield from walk(j + 1, used + (k,))
    yield from walk(0, ())
