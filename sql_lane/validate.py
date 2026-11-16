"""Validate: checks the result Execute returned (step2-sql-lane-rules.md, B5).

Review checks the SQL before it runs; Validate checks the rows after. It catches results
that obviously don't fit the question (one value expected but 50 rows, "top 5" with 1,000
rows, a column that is NULL everywhere, a percentage of 4,350) and sends the reason back to
Generate SQL. Plain code: no LLM, no database. It does not check that the numbers are
correct; the evaluation set does that.

Order (the first failure is the only one recorded):
  0 empty result (0 rows, or every value NULL): the special empty rule below
  1 shape        2 row cap        3 all-NULL column        4 ordering        5 impossible values

Empty rule: the first empty result is retried once (stage "empty", empty_retry_done = True).
An empty result after that retry, or on the last attempt, is accepted as a real "no matching
data" answer: status "ok". Every value NULL counts as empty because SUM/AVG over no
matching rows returns one row of NULL, not zero rows (project owner's decision).
"""
import sqlglot
from sqlglot import exp

from sql_lane.state import SqlLaneState

MAX_ATTEMPTS = 3

EMPTY_MESSAGE = ("Returned no data (0 rows, or only NULL values). Check filter values "
                 "against the catalog values and date ranges.")

# Column-name endings (SQL rule 4) and the values they can't have.
MONEY_ENDINGS = ("revenue", "price", "amount", "freight")
NON_NEGATIVE_ENDINGS = MONEY_ENDINGS + ("_count", "units", "_days")
SIGNED_ENDINGS = ("_change_pct", "_growth_pct", "_diff_days")   # may be negative / above 100


def is_empty(rows: list[list]) -> bool:
    """No rows, or rows where every value is NULL (an aggregate over nothing)."""
    return not rows or all(v is None for row in rows for v in row)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_shape(columns, rows, shape) -> str | None:
    """1. single_value = exactly one row and one column."""
    if shape["kind"] == "single_value" and (len(rows) > 1 or len(columns) > 1):
        return (f"Expected a single value but got {len(rows)} rows / {len(columns)} columns. "
                "Return one row and one column, or use kind 'list' with max_rows.")
    return None


def _check_row_cap(rows, truncated, shape) -> str | None:
    """2. No more rows than max_rows; a single value is never truncated."""
    max_rows = shape.get("max_rows")
    if max_rows is not None and len(rows) > max_rows:
        return f"Expected at most {max_rows} rows but got {len(rows)}; add LIMIT or aggregate."
    if truncated and shape["kind"] == "single_value":
        return "Expected a single value but got more than 1,000 rows; aggregate."
    return None


def _check_all_null(columns, rows) -> str | None:
    """3. No column that is NULL in every row (usually a wrong join or filter)."""
    for i, name in enumerate(columns):
        if all(row[i] is None for row in rows):
            return f"Column {name} is NULL in every row; check joins and filters."
    return None


def _check_ordering(sql, shape) -> str | None:
    """4. If the order matters, the outer query has ORDER BY (checked on the parsed SQL)."""
    if not shape.get("order_matters"):
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return None                         # Review parsed it; never block on a parser quirk
    if not tree.args.get("order"):
        return "The question needs an order; add ORDER BY to the outer query."
    return None


def _check_values(columns, rows) -> str | None:
    """5. Values a column with this name ending can't have."""
    for i, name in enumerate(columns):
        n = name.lower()
        if n.endswith(SIGNED_ENDINGS):
            continue
        for row in rows:
            v = row[i]
            if not _is_number(v):
                continue
            bad = ((n.endswith(NON_NEGATIVE_ENDINGS) and v < 0)
                   or (n.endswith("score") and not 1 <= v <= 5)
                   or (n.endswith("_pct") and not 0 <= v <= 100))
            if bad:
                return f"Column {name} has impossible value {v}."
    return None


def check_result(columns: list[str], rows: list[list], truncated: bool,
                 expected_shape: dict, sql: str) -> str | None:
    """Checks 1-5 on a non-empty result. Returns the first failure's message, or None."""
    return (_check_shape(columns, rows, expected_shape)
            or _check_row_cap(rows, truncated, expected_shape)
            or _check_all_null(columns, rows)
            or _check_ordering(sql, expected_shape)
            or _check_values(columns, rows))


def validate(state: SqlLaneState) -> dict:
    """The Validate node: the empty rule first, then check_result."""
    rows = state.get("rows", [])
    attempt = len(state.get("sql", []))

    if is_empty(rows):
        if state.get("empty_retry_done") or attempt >= MAX_ATTEMPTS:
            return {"status": "ok"}         # a real "no matching data" answer
        return {"errors": [{"stage": "empty", "message": EMPTY_MESSAGE}],
                "empty_retry_done": True}

    error = check_result(state.get("columns", []), rows, state.get("truncated", False),
                         state["expected_shape"], state["sql"][-1])
    if error:
        return {"errors": [{"stage": "validate", "message": error}]}
    return {"status": "ok"}
