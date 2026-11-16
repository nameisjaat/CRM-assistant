"""Load the Olist CSVs into Postgres as raw.<table>, using the table names from
step1-data-model-olist.md ("Source files -> raw tables").

Every column is stored as TEXT so the raw layer is a copy of the CSVs;
types are applied later when cleaning into the analytics schema.
Files are read as utf-8-sig so a byte-order mark is not glued to the first header.

Usage:
    python load_raw.py                 # reload every raw table
    python load_raw.py products leads  # reload only the named raw tables
Re-running drops and reloads each table it loads.
"""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

ROOT = Path(__file__).resolve().parent

# raw table name -> CSV path (relative to the project root)
SOURCES = {
    "geolocation": "olist_public_dataset/olist_geolocation_dataset.csv",
    "customers": "olist_public_dataset/olist_customers_dataset.csv",
    "orders": "olist_public_dataset/olist_orders_dataset.csv",
    "order_items": "olist_public_dataset/olist_order_items_dataset.csv",
    "order_payments": "olist_public_dataset/olist_order_payments_dataset.csv",
    "order_reviews": "olist_public_dataset/olist_order_reviews_dataset.csv",
    "products": "olist_public_dataset/olist_products_dataset.csv",
    "sellers": "olist_public_dataset/olist_sellers_dataset.csv",
    "product_category_name_translation": "olist_public_dataset/product_category_name_translation.csv",
    "leads": "olist_marketing_dataset/olist_marketing_qualified_leads_dataset.csv",
    "closed_deals": "olist_marketing_dataset/olist_closed_deals_dataset.csv",
}


def load_table(conn, table, csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        columns = [c.strip().strip('"') for c in f.readline().strip().split(",")]
    # reopen so COPY reads from the start (utf-8-sig skips the BOM again)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        cols = [sql.Identifier(c) for c in columns]
        with conn, conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS raw.{}").format(sql.Identifier(table)))
            cur.execute(
                sql.SQL("CREATE TABLE raw.{} ({})").format(
                    sql.Identifier(table),
                    sql.SQL(", ").join(sql.SQL("{} TEXT").format(c) for c in cols),
                )
            )
            cur.copy_expert(
                sql.SQL(
                    "COPY raw.{} FROM STDIN WITH (FORMAT csv, HEADER true, FORCE_NULL ({}))"
                ).format(sql.Identifier(table), sql.SQL(", ").join(cols)).as_string(conn),
                f,
            )
            cur.execute(sql.SQL("SELECT count(*) FROM raw.{}").format(sql.Identifier(table)))
            print(f"raw.{table}: {cur.fetchone()[0]} rows")


def main():
    requested = sys.argv[1:] or list(SOURCES)
    unknown = [t for t in requested if t not in SOURCES]
    if unknown:
        sys.exit(f"Unknown raw table(s): {', '.join(unknown)}. Known: {', '.join(SOURCES)}")

    load_dotenv(ROOT / ".env")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS raw")
        for table in requested:
            load_table(conn, table, ROOT / SOURCES[table])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
