"""Build the schema catalog from the analytics schema in Postgres.

    python catalog/build_catalog.py

Writes catalog/schema_catalog.json. It is used by schema select (which tables and
columns go into the SQL-generation prompt) and by the review step (known tables and
columns). The descriptions come from the COMMENTs in db/transform.sql, which stay the
single source of truth: after changing transform.sql and rebuilding analytics, run
this again. Nothing in the JSON is edited by hand.

What it reads (read-only transaction, nothing is changed):
- every table and column in `analytics`, with types and comments
- primary keys and foreign keys
- exact row counts
- for low-cardinality text columns: every distinct value, most common first, so the
  LLM writes category = 'health_beauty' instead of guessing
- min/max for date, timestamp and numeric columns, so the LLM knows the data period
- data_facts: the date range of orders and of the lead funnel, as a top-level section
  that is always sent to SQL generation, whichever tables are selected

The hand-written parts are SOFT_JOINS below (joins that deliberately have no foreign
key, because some child rows have no parent, so they can't be read from Postgres) and
DATA_FACTS (which date ranges to read and what each one means; the dates themselves
are read from the database).

Connection string: DATABASE_URL (from the environment or the project's .env).
"""
import json
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "schema_catalog.json"
SCHEMA = "analytics"

# Text columns with at most this many distinct values get their values listed.
MAX_LISTED_VALUES = 100

# Joins with no foreign key, by design (see "Join rules" in step1-data-model-olist.md).
# Always LEFT JOIN: some child rows have no match.
SOFT_JOINS = [
    {"from": "customers.zip_prefix", "to": "geolocation.zip_prefix",
     "note": "Some customer zips are missing from geolocation."},
    {"from": "sellers.zip_prefix", "to": "geolocation.zip_prefix",
     "note": "Some seller zips are missing from geolocation."},
    {"from": "orders.delivery_zip_prefix", "to": "geolocation.zip_prefix",
     "note": "Some delivery zips are missing from geolocation."},
    {"from": "closed_deals.seller_id", "to": "sellers.seller_id",
     "note": "Many deal sellers never sold anything, so they have no row in sellers."},
]

# Date ranges always given to SQL generation (step2-sql-lane-rules.md, B1 rule 7), so it
# anchors "this year" / "last month" to the data instead of today's date, and recognizes
# questions about periods with no data. Each entry: (table, column, what the range means).
DATA_FACTS = [
    ("orders", "purchased_at",
     "Period covered by orders. 'This year', 'last month', 'recently' are anchored to the "
     "latest date here, not today's date. Nothing exists after it."),
    ("leads", "first_contact_date",
     "Period covered by the marketing funnel (leads and closed deals). Funnel questions "
     "outside it have no data."),
]

TEXT_TYPES = {"text", "character varying", "character"}
RANGE_TYPES = ("date", "timestamp", "numeric", "integer", "bigint", "smallint",
               "double precision", "real")


def fetch(cur, query, params=None):
    cur.execute(query, params)
    return cur.fetchall()


def list_tables(cur):
    return fetch(cur, """
        SELECT c.relname, obj_description(c.oid, 'pg_class')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relkind = 'r'
        ORDER BY c.relname""", (SCHEMA,))


def list_columns(cur, table):
    return fetch(cur, """
        SELECT a.attname,
               format_type(a.atttypid, a.atttypmod),
               col_description(a.attrelid, a.attnum),
               a.attgenerated <> ''
        FROM pg_attribute a
        WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum""", (f"{SCHEMA}.{table}",))


def primary_key(cur, table):
    rows = fetch(cur, """
        SELECT a.attname
        FROM pg_constraint c
        JOIN unnest(c.conkey) WITH ORDINALITY k(attnum, pos) ON true
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
        WHERE c.conrelid = %s::regclass AND c.contype = 'p'
        ORDER BY k.pos""", (f"{SCHEMA}.{table}",))
    return [r[0] for r in rows]


def foreign_keys(cur):
    rows = fetch(cur, """
        SELECT src.relname, sa.attname, dst.relname, da.attname
        FROM pg_constraint c
        JOIN pg_class src ON src.oid = c.conrelid
        JOIN pg_class dst ON dst.oid = c.confrelid
        JOIN unnest(c.conkey, c.confkey) AS k(src_att, dst_att) ON true
        JOIN pg_attribute sa ON sa.attrelid = c.conrelid  AND sa.attnum = k.src_att
        JOIN pg_attribute da ON da.attrelid = c.confrelid AND da.attnum = k.dst_att
        WHERE c.contype = 'f' AND c.connamespace = %s::regnamespace
        ORDER BY 1, 2""", (SCHEMA,))
    return [{"from": f"{s}.{sc}", "to": f"{d}.{dc}", "type": "foreign_key"}
            for s, sc, d, dc in rows]


def describe_column(cur, table, name, col_type):
    """Distinct values for small text columns; min/max for dates and numbers."""
    ident = sql.Identifier(name)
    tbl = sql.Identifier(SCHEMA, table)
    extra = {}
    if col_type in TEXT_TYPES and not name.endswith("_id"):
        cur.execute(sql.SQL("SELECT count(DISTINCT {}) FROM {}").format(ident, tbl))
        distinct = cur.fetchone()[0]
        extra["distinct_values"] = distinct
        if distinct <= MAX_LISTED_VALUES:
            cur.execute(sql.SQL(
                "SELECT {c} FROM {t} WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY count(*) DESC, {c}"
            ).format(c=ident, t=tbl))
            extra["values"] = [r[0] for r in cur.fetchall()]
    elif col_type.startswith(RANGE_TYPES):
        cur.execute(sql.SQL("SELECT min({c}), max({c}) FROM {t}").format(c=ident, t=tbl))
        lo, hi = cur.fetchone()
        if lo is not None:
            extra["min"], extra["max"] = str(lo), str(hi)
    return extra


def build(cur):
    tables, missing_comments = [], []
    for table, table_comment in list_tables(cur):
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(SCHEMA, table)))
        row_count = cur.fetchone()[0]
        pk = primary_key(cur, table)
        columns = []
        for name, col_type, comment, generated in list_columns(cur, table):
            if not comment:
                missing_comments.append(f"{table}.{name}")
            col = {"name": name, "type": col_type, "description": comment or ""}
            if name in pk:
                col["primary_key"] = True
            if generated:
                col["generated"] = True
            col.update(describe_column(cur, table, name, col_type))
            columns.append(col)
        if not table_comment:
            missing_comments.append(table)
        tables.append({"name": table, "description": table_comment or "",
                       "row_count": row_count, "primary_key": pk, "columns": columns})

    known = {f"{t['name']}.{c['name']}" for t in tables for c in t["columns"]}
    for j in SOFT_JOINS:
        for side in (j["from"], j["to"]):
            if side not in known:
                sys.exit(f"SOFT_JOINS refers to {side}, which is not in {SCHEMA}. Update SOFT_JOINS.")

    joins = foreign_keys(cur) + [{**j, "type": "left_join_only"} for j in SOFT_JOINS]
    return {"schema": SCHEMA, "tables": tables, "joins": joins,
            "data_facts": data_facts(cur, known)}, missing_comments


def data_facts(cur, known):
    """Earliest and latest value of each DATA_FACTS column, read from the database."""
    facts = []
    for table, column, note in DATA_FACTS:
        if f"{table}.{column}" not in known:
            sys.exit(f"DATA_FACTS refers to {table}.{column}, which is not in {SCHEMA}. Update DATA_FACTS.")
        cur.execute(sql.SQL("SELECT min({c}), max({c}) FROM {t}").format(
            c=sql.Identifier(column), t=sql.Identifier(SCHEMA, table)))
        lo, hi = cur.fetchone()
        facts.append({"table": table, "column": column,
                      "earliest": str(lo), "latest": str(hi), "note": note})
    return facts


def main():
    load_dotenv(ROOT / ".env")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            # Same zone as crm_readonly, so timestamp ranges match what queries will return.
            cur.execute("SET TIME ZONE 'America/Sao_Paulo'")
            catalog, missing = build(cur)
    finally:
        conn.rollback()
        conn.close()

    text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    OUT.write_text(text, encoding="utf-8")

    n_cols = sum(len(t["columns"]) for t in catalog["tables"])
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(catalog['tables'])} tables, {n_cols} columns, "
          f"{len(catalog['joins'])} joins, {len(catalog['data_facts'])} data facts, "
          f"{len(text):,} characters.")
    if missing:
        print(f"WARNING: {len(missing)} without a comment: {', '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
