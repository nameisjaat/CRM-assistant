"""Smoke test: run schema select on one question and print what it picked.

    .venv\\Scripts\\python.exe tests/sql_lane/test_schema_select.py

Calls OpenAI (gpt-4o-mini) once. Reads OPENAI_API_KEY from the project's .env.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # so `sql_lane` imports when run as a script

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from sql_lane.schema_select import finalize_tables, make_schema_select, select_tables

QUESTION = "What is the average review score by product category?"


def test_schema_select():
    load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. The LLM's own choice
    selection = select_tables(QUESTION, llm)
    print(f"\nQuestion:      {QUESTION}")
    print(f"LLM reason:    {selection.reason}")
    print(f"LLM picked:    {selection.tables}")
    print(f"After rules:   {finalize_tables(selection.tables)}")

    # 2. The full node, as the graph will run it (makes its own LLM call)
    out = make_schema_select(llm)({"question": QUESTION})
    print(f"\nNode tables:   {out['tables']}")
    print("Schema context:")
    for name, entry in out["schema_context"].items():
        joins = [f"{j['from']} = {j['to']} ({j['type']})" for j in entry["joins"]]
        print(f"  {name}: {len(entry['columns'])} columns, {entry['row_count']:,} rows, "
              f"key {entry['primary_key']}")
        for j in joins:
            print(f"      join {j}")

    assert out["tables"], "schema select returned no tables"


if __name__ == "__main__":
    test_schema_select()
