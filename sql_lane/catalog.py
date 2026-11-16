"""Read-only access to catalog/schema_catalog.json for the sql_query lane.

The catalog is built from the database by catalog/build_catalog.py; this module never
changes it. It is loaded once and reused by schema select now, and by review and
generate SQL later.
"""
import copy
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "schema_catalog.json"

# Linked to customers, sellers and orders only by zip area. Never used as an in-between
# table: joining sellers to customers "through" a zip area would be a wrong join.
GEOLOCATION = "geolocation"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """The whole catalog, read once. Treat as read-only (copy before changing anything)."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def table_names() -> list[str]:
    """All table names, in catalog order."""
    return [t["name"] for t in load_catalog()["tables"]]


def _table_of(side: str) -> str:
    return side.split(".", 1)[0]          # "orders.customer_id" -> "orders"


def _neighbours() -> dict[str, list[str]]:
    """Which tables join directly, from the catalog's joins (foreign keys and LEFT JOIN-only)."""
    order = table_names()
    adj = {name: set() for name in order}
    for j in load_catalog()["joins"]:
        a, b = _table_of(j["from"]), _table_of(j["to"])
        adj[a].add(b)
        adj[b].add(a)
    # sorted in catalog order, so ties in the path search are always broken the same way
    return {name: sorted(adj[name], key=order.index) for name in order}


def _shortest_path(sources: list[str], target: str, avoid: set[str]) -> list[str]:
    """Tables on the shortest join path from any table in `sources` to `target`.

    Breadth-first search; tables in `avoid` are never stepped through. Returns the path
    including both ends, or [] if the target can't be reached.
    """
    neighbours = _neighbours()
    came_from = {s: None for s in sources}
    queue = deque(sources)
    while queue:
        current = queue.popleft()
        if current == target:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for nxt in neighbours[current]:
            if nxt in came_from or (nxt in avoid and nxt != target):
                continue
            came_from[nxt] = current
            queue.append(nxt)
    return []


def connect_tables(tables: list[str]) -> list[str]:
    """Add the in-between tables needed to join `tables` together.

    Example: ["customers", "order_items"] -> ["customers", "order_items", "orders"].
    Tables are connected in the order given, each by the shortest path to the ones
    already connected. geolocation is never an in-between table: the other tables are
    connected without it, and if it was asked for, it is attached last.
    Returns the given tables first (in their order), then the added ones.
    """
    result: list[str] = []
    core = [t for t in tables if t != GEOLOCATION]
    for table in core:
        if table in result:
            continue
        if not result:
            result.append(table)
            continue
        path = _shortest_path(result, table, avoid={GEOLOCATION})
        result.extend(t for t in path if t not in result)

    if GEOLOCATION in tables:
        if result:
            path = _shortest_path(result, GEOLOCATION, avoid=set())
            result.extend(t for t in path if t not in result)
        else:
            result.append(GEOLOCATION)

    # keep the caller's tables first, in their order, then the added in-between tables
    given = [t for t in dict.fromkeys(tables) if t in result]
    return given + [t for t in result if t not in given]


def build_schema_context(tables: list[str]) -> dict[str, dict]:
    """Everything the catalog has about each table in `tables`, keyed by table name.

    Each entry is the table's full catalog entry (description, row_count, primary_key,
    and every column with its type, description, and when present its allowed values,
    distinct count, min/max, primary_key and generated flags), plus `joins`: the
    catalog's joins between this table and the other tables in `tables` (type
    "foreign_key" or "left_join_only"). Joins to tables that were not selected are left
    out, so the prompt never points at a table the SQL may not use.
    """
    catalog = load_catalog()
    by_name = {t["name"]: t for t in catalog["tables"]}
    selected = set(tables)
    context = {}
    for name in tables:
        entry = copy.deepcopy(by_name[name])
        entry.pop("name", None)                       # already the key
        entry["joins"] = [
            copy.deepcopy(j) for j in catalog["joins"]
            if name in (_table_of(j["from"]), _table_of(j["to"]))
            and {_table_of(j["from"]), _table_of(j["to"])} <= selected
        ]
        context[name] = entry
    return context
