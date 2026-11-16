"""Review, layer 1: every deterministic check on the SQL (step2-sql-lane-rules.md, B3).

code_review(sql) parses the SQL once with sqlglot and runs the checks below in order. The
first failure is returned as feedback for Generate SQL; None means every check passed.

Safety (final: SQL failing these never reaches the database or the LLM reviewer)
   1 parses, exactly one statement        4 only known columns
   2 read-only                            5 no dangerous functions
   3 only known tables, analytics schema  6 no %(name)s placeholders
Data model (global rules that can be checked from the SQL alone)
   7 C5   no row multiplication           12 C6   average score filters is_latest_for_order
   8 SQL3 no SELECT *                     13 C7   delivery filters delivered, NOT has_date_issue
   9 C10  geolocation is LEFT JOIN        14 SQL5 no round() inside an aggregate
  10 C12  behaviour profile uses ILIKE    15 SQL4 _pct columns are 0-100
  11 C1   revenue: unit_price only, canceled and unavailable excluded

The data-model checks fire from what the SQL itself does (it averages a score, it uses a
delivery column, ...), so they need no understanding of the question. A required filter
counts when it is in a SELECT connected to the one that triggers the rule (the same
SELECT, a CTE or subquery it reads, or a SELECT that reads it), not in an unrelated CTE.
Never rewrites the SQL.
"""
import re
from functools import lru_cache
from typing import Callable

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope, walk_in_scope

from sql_lane.catalog import load_catalog, table_names

DIALECT = "postgres"
ALLOWED_SCHEMA = "analytics"

# Statements that change data, schema, permissions or settings. Command is what sqlglot
# returns for syntax it can't classify (CALL, DO, ...), so it is rejected too.
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Alter,
                   exp.Create, exp.TruncateTable, exp.Grant, exp.Copy, exp.Set, exp.Command)

DANGEROUS_FUNCTIONS = {"pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
                       "lo_import", "lo_export", "dblink", "set_config", "current_setting",
                       "pg_terminate_backend", "pg_cancel_backend"}

# One-to-many children of orders: two in one SELECT multiply rows (global rule 5).
ORDER_CHILDREN = {"order_items", "order_payments", "order_reviews"}

EXCLUDED_FROM_REVENUE = {"canceled", "unavailable"}
DELIVERY_COLUMNS = {"delivery_days", "is_late"}


# ================================================================ helpers

@lru_cache(maxsize=1)
def _column_schema() -> dict[str, dict[str, str]]:
    """{table: {column: type}} from the catalog, the form sqlglot's qualify expects."""
    return {t["name"]: {c["name"]: c["type"] for c in t["columns"]}
            for t in load_catalog()["tables"]}


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name for cte in tree.find_all(exp.CTE)}


def _real_tables(tree: exp.Expression) -> list[exp.Table]:
    """Table references that name a table (not a table function like generate_series)."""
    return [t for t in tree.find_all(exp.Table) if isinstance(t.this, exp.Identifier)]


def _has_column(node: exp.Expression, names: set[str]) -> bool:
    return any(c.name in names for c in node.find_all(exp.Column))


def _string(node: exp.Expression) -> str | None:
    return node.this if isinstance(node, exp.Literal) and node.is_string else None


class _Scopes:
    """The SELECTs of a query (sqlglot scopes) and how they connect.

    Two SELECTs are connected when one reads the other (a CTE or subquery in its FROM or
    JOIN, or a subquery in its WHERE). `connected(node)` is the group of SELECTs whose rows
    can reach `node`'s SELECT or be reached from it; a filter anywhere in that group applies.
    """

    def __init__(self, tree: exp.Expression):
        self.scopes = traverse_scope(tree)
        self.owner: dict[int, Scope] = {}
        for s in self.scopes:
            for node in walk_in_scope(s.expression):
                self.owner.setdefault(id(node), s)
        self.edges: dict[int, set[int]] = {id(s): set() for s in self.scopes}
        for s in self.scopes:
            linked = [src for _, src in s.selected_sources.values() if isinstance(src, Scope)]
            linked += list(s.subquery_scopes) + list(getattr(s, "set_operation_scopes", None) or getattr(s, "union_scopes", []))
            for other in linked:
                self.edges[id(s)].add(id(other))
                self.edges.setdefault(id(other), set()).add(id(s))
        self.by_id = {id(s): s for s in self.scopes}

    def connected(self, node: exp.Expression) -> list[Scope]:
        start = self.owner.get(id(node))
        if start is None:
            return self.scopes
        seen, todo = {id(start)}, [id(start)]
        while todo:
            for nxt in self.edges.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return [self.by_id[i] for i in seen if i in self.by_id]

    @staticmethod
    def conditions(scopes: list[Scope]) -> list[exp.Expression]:
        """The WHERE, JOIN ... ON and HAVING conditions of these SELECTs."""
        out = []
        for s in scopes:
            e = s.expression
            if not isinstance(e, exp.Select):
                continue
            for key in ("where", "having"):
                if e.args.get(key):
                    out.append(e.args[key].this)
            for j in e.args.get("joins") or []:
                if j.args.get("on"):
                    out.append(j.args["on"])
        return out


def _status_filter(conditions: list[exp.Expression]) -> tuple[set[str], set[str]]:
    """(statuses required, statuses excluded) by filters on a `status` column."""
    required, excluded = set(), set()
    for cond in conditions:
        for node in cond.find_all(exp.EQ, exp.NEQ, exp.In):
            col = node.this if isinstance(node.this, exp.Column) else None
            if col is None or col.name != "status":
                continue
            negated = isinstance(node.parent, exp.Not)
            if isinstance(node, exp.In):
                values = {v for v in map(_string, node.expressions) if v}
            else:
                values = {v for v in [_string(node.expression)] if v}
            if isinstance(node, exp.NEQ) or negated:
                excluded |= values
            else:
                required |= values
    return required, excluded


def _excludes_date_issues(conditions: list[exp.Expression]) -> bool:
    """True if some condition keeps only rows where has_date_issue is false."""
    for cond in conditions:
        for col in cond.find_all(exp.Column):
            if col.name != "has_date_issue":
                continue
            parent = col.parent
            if isinstance(parent, exp.Not):                                   # NOT has_date_issue
                return True
            if isinstance(parent, (exp.EQ, exp.Is)) and isinstance(parent.expression, exp.Boolean) \
                    and parent.expression.this is False:                     # = false / IS FALSE
                return True
            if isinstance(parent, exp.Is) and isinstance(parent.parent, exp.Not) \
                    and isinstance(parent.expression, exp.Boolean) and parent.expression.this is True:
                return True                                                   # IS NOT TRUE
    return False


def _keeps_latest_review(conditions: list[exp.Expression]) -> bool:
    """True if some condition filters on is_latest_for_order (not negated)."""
    for cond in conditions:
        for col in cond.find_all(exp.Column):
            if col.name == "is_latest_for_order" and not isinstance(col.parent, exp.Not):
                return True
    return False


# ================================================================ safety checks

def _check_statement(sql: str) -> tuple[exp.Expression | None, str | None]:
    """1. Parses, exactly one statement. A trailing semicolon is fine."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as e:
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(e))   # drop sqlglot's terminal colours
        return None, f"SQL could not be parsed: {message}"
    if len(statements) != 1:
        return None, f"Only one statement is allowed; found {len(statements)}."
    return statements[0], None


def _check_read_only(tree: exp.Expression) -> str | None:
    """2. A SELECT or WITH query, with nothing that writes anywhere inside it."""
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return ("Only read-only SELECT queries are allowed; found "
                    f"{type(node).__name__.upper()}.")
        if isinstance(node, exp.Into):
            return "Only read-only SELECT queries are allowed; SELECT ... INTO is not."
    if not isinstance(tree, exp.Query):     # Select, Union/Intersect/Except; WITH sits inside
        return f"The query must start with SELECT or WITH; found {type(tree).__name__.upper()}."
    return None


def _check_tables(tree: exp.Expression) -> str | None:
    """3. Only catalog tables or CTE names, and no schema but analytics."""
    known, ctes = set(table_names()), _cte_names(tree)
    for t in _real_tables(tree):
        if t.catalog or (t.db and t.db != ALLOWED_SCHEMA):
            return f"Only the {ALLOWED_SCHEMA} schema is allowed; found {t.sql(dialect=DIALECT)}."
        if t.name not in known and t.name not in ctes:
            return (f"Unknown table {t.name}. Use only these tables: "
                    f"{', '.join(sorted(known))}.")
    return None


def _check_columns(tree: exp.Expression) -> str | None:
    """4. Every column exists, using sqlglot's qualify with the catalog's columns.

    A real "unknown column" error (OptimizeError) is returned as feedback. Any other error
    is sqlglot not understanding valid Postgres; then the check is skipped and the
    database's own error in Execute catches a real mistake.
    """
    copy = tree.copy()
    for t in _real_tables(copy):            # the schema has no 'analytics' level
        if t.db == ALLOWED_SCHEMA:
            t.set("db", None)
    try:
        qualify(copy, schema=_column_schema(), dialect=DIALECT, validate_qualify_columns=True)
    except OptimizeError as e:
        return f"Column check failed: {e}"
    except Exception:                       # sqlglot limitation, not a SQL mistake
        return None
    return None


def _check_functions(tree: exp.Expression) -> str | None:
    """5. None of the dangerous Postgres functions."""
    for f in tree.find_all(exp.Func):
        name = (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
        if name in DANGEROUS_FUNCTIONS:
            return f"Function {name} is not allowed."
    return None


def _check_placeholders(tree: exp.Expression) -> str | None:
    """6. No %(name)s parameters (none are provided in this version)."""
    if any(True for _ in tree.find_all(exp.Placeholder, exp.Parameter)):
        return "Parameters are not available; write values directly in the SQL."
    return None


# ================================================================ data-model checks

def _check_row_multiplication(tree: exp.Expression) -> str | None:
    """7. (C5) No SELECT reads two of order_items/order_payments/order_reviews directly."""
    ctes = _cte_names(tree)
    for select in tree.find_all(exp.Select):
        sources = []
        from_ = select.args.get("from_") or select.args.get("from")
        if from_:
            sources.append(from_.this)
        sources += [j.this for j in select.args.get("joins") or []]
        direct = {s.name for s in sources
                  if isinstance(s, exp.Table) and isinstance(s.this, exp.Identifier)
                  and s.name not in ctes}
        clash = sorted(direct & ORDER_CHILDREN)
        if len(clash) >= 2:
            return (f"Rule C5 broken: one SELECT joins {' and '.join(clash)}. Aggregate each of "
                    "these tables to one row per order_id in its own CTE before joining them.")
    return None


def _check_star(tree: exp.Expression) -> str | None:
    """8. (SQL3) No SELECT * or t.* in any select list (COUNT(*) is fine)."""
    for select in tree.find_all(exp.Select):
        for e in select.expressions:
            if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
                return "Name every column; SELECT * is not allowed."
    return None


def _check_geolocation_join(tree: exp.Expression) -> str | None:
    """9. (C10) Every JOIN to geolocation is a LEFT JOIN."""
    for j in tree.find_all(exp.Join):
        if isinstance(j.this, exp.Table) and j.this.name == "geolocation" and j.side != "LEFT":
            return ("Rule C10 broken: join geolocation with LEFT JOIN; some zip prefixes have "
                    "no geolocation row.")
    return None


def _check_behaviour_profile(tree: exp.Expression) -> str | None:
    """10. (C12) behaviour_profile is matched with ILIKE, never = / <> / IN."""
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.In):
        if isinstance(node.this, exp.Column) and node.this.name == "behaviour_profile":
            return ("Rule C12 broken: profiles can be combined ('cat, wolf'); match "
                    "behaviour_profile with ILIKE '%wolf%', not with = or IN.")
    return None


def _check_revenue(tree: exp.Expression, scopes: _Scopes) -> str | None:
    """11. (C1) A column named like revenue is SUM(unit_price) without freight or payments,
    over orders that exclude 'canceled' and 'unavailable' (or keep only other statuses)."""
    for alias in tree.find_all(exp.Alias):
        if "revenue" not in alias.alias.lower() or not alias.this.find(exp.Sum):
            continue
        if _has_column(alias.this, {"amount"}):
            return ("Rule C1 broken: revenue must be SUM(order_items.unit_price), never "
                    "computed from order_payments.amount.")
        if _has_column(alias.this, {"freight", "item_total"}):
            return ("Rule C1 broken: revenue is SUM(order_items.unit_price) only; freight "
                    "(and item_total, which includes freight) is separate.")
        if not _has_column(alias.this, {"unit_price"}):
            continue
        required, excluded = _status_filter(_Scopes.conditions(scopes.connected(alias)))
        ok = (required and not required & EXCLUDED_FROM_REVENUE) or EXCLUDED_FROM_REVENUE <= excluded
        if not ok:
            return ("Rule C1 broken: revenue must exclude orders whose status is 'canceled' or "
                    "'unavailable': add WHERE status NOT IN ('canceled', 'unavailable') on "
                    "orders in the query that computes it.")
    return None


def _check_review_score(tree: exp.Expression, scopes: _Scopes) -> str | None:
    """12. (C6) An average of score keeps only order_reviews WHERE is_latest_for_order."""
    for avg in tree.find_all(exp.Avg):
        if _has_column(avg, {"score"}) and \
                not _keeps_latest_review(_Scopes.conditions(scopes.connected(avg))):
            return ("Rule C6 broken: the average review score must use only "
                    "order_reviews WHERE is_latest_for_order, so each order counts once.")
    return None


def _check_delivery(tree: exp.Expression, scopes: _Scopes) -> str | None:
    """13. (C7) Using delivery_days or is_late requires status = 'delivered' AND NOT has_date_issue."""
    for col in tree.find_all(exp.Column):
        if col.name not in DELIVERY_COLUMNS:
            continue
        conditions = _Scopes.conditions(scopes.connected(col))
        required, _ = _status_filter(conditions)
        if required != {"delivered"} or not _excludes_date_issues(conditions):
            return (f"Rule C7 broken: queries using {col.name} must filter "
                    "status = 'delivered' AND NOT has_date_issue.")
    return None


def _check_round_in_aggregate(tree: exp.Expression) -> str | None:
    """14. (SQL5) round() is applied after aggregating, never inside SUM/AVG/..."""
    for agg in tree.find_all(exp.AggFunc):
        if agg.find(exp.Round):
            return (f"Rule SQL5 broken: round after aggregating, e.g. round({agg.key}(x), 2), "
                    f"not {agg.key}(round(x, 2)): {agg.sql(dialect=DIALECT)}")
    return None


def _check_pct(tree: exp.Expression) -> str | None:
    """15. (SQL4) A column ending in _pct runs 0-100, so its expression multiplies by 100."""
    for alias in tree.find_all(exp.Alias):
        if not alias.alias.lower().endswith("_pct"):
            continue
        if isinstance(alias.this, exp.Column):          # passing through an already computed _pct
            continue
        hundred = any(lit.is_number and float(lit.this) == 100
                      for lit in alias.this.find_all(exp.Literal))
        if not hundred:
            return (f"Rule SQL4 broken: {alias.alias} must be a percentage from 0 to 100; "
                    "multiply the ratio by 100.")
    return None


# ================================================================ entry point

TREE_CHECKS: list[Callable[[exp.Expression], str | None]] = [
    _check_read_only, _check_tables, _check_columns, _check_functions, _check_placeholders,
    _check_row_multiplication, _check_star, _check_geolocation_join, _check_behaviour_profile,
]
SCOPE_CHECKS: list[Callable[[exp.Expression, _Scopes], str | None]] = [
    _check_revenue, _check_review_score, _check_delivery,
]
FINAL_CHECKS: list[Callable[[exp.Expression], str | None]] = [
    _check_round_in_aggregate, _check_pct,
]


def code_review(sql: str) -> str | None:
    """Run checks 1-15 in order. Returns the first failure's message, or None if all pass."""
    tree, error = _check_statement(sql)
    if error:
        return error
    for check in TREE_CHECKS:
        if error := check(tree):
            return error
    try:
        scopes = _Scopes(tree)
    except Exception:                       # sqlglot can't build scopes: skip these three
        scopes = None
    if scopes is not None:
        for check in SCOPE_CHECKS:
            if error := check(tree, scopes):
                return error
    for check in FINAL_CHECKS:
        if error := check(tree):
            return error
    return None
