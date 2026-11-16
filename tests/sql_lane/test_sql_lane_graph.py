"""Smoke test: run the whole sql_query lane on three questions and print what happened.

    .venv\\Scripts\\python.exe tests\\sql_lane\\test_sql_lane_graph.py

Calls OpenAI (gpt-4o-mini for schema select and review, gpt-4o for generate) and queries the Neon database (read-only).
Reads OPENAI_API_KEY and DATABASE_URL from the project's .env.

The three questions expect different shapes of answer:
  1. one number                    (single_value)
  2. a ranked top-N list           (list, max_rows 5, ordered)
  3. one row per month             (time_series, ordered by month)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # so `sql_lane` imports when run as a script

from dotenv import load_dotenv

from sql_lane.graph import build_sql_lane

QUESTIONS = [
    "How many orders were delivered late in 2018?",
    "Which 5 product categories had the most revenue in 2018?",
    "What was the average delivery time in days for each month of 2018?",
]

MAX_ROWS_SHOWN = 15


def print_result(question: str, result: dict, seconds: float) -> None:
    line = "=" * 90
    print(f"\n{line}\nQuestion: {question}\n{line}")
    print(f"Status:      {result.get('status')}   ({seconds:.1f}s, "
          f"{len(result.get('sql', []))} attempt(s))")
    print(f"Tables:      {result.get('tables')}")
    print(f"Shape:       {result.get('expected_shape')}")

    if result.get("unanswerable_reason"):
        print(f"Unanswerable: {result['unanswerable_reason']}")

    print("Assumptions:")
    for a in result.get("assumptions", []) or ["(none)"]:
        print(f"  - {a}")

    # every attempt with the error it caused (the last one has none if it succeeded)
    errors = result.get("errors", [])
    for i, sql in enumerate(result.get("sql", []), 1):
        print(f"\n--- Attempt {i} SQL ---\n{sql or '(no usable SQL)'}")
        if i <= len(errors):
            e = errors[i - 1]
            print(f"--- Attempt {i} failed at {e['stage']}: {e['message']}")

    columns, rows = result.get("columns", []), result.get("rows", [])
    print(f"\nResult: {len(rows)} row(s){' (truncated at 1,000)' if result.get('truncated') else ''}")
    if columns:
        print("  " + " | ".join(columns))
        for row in rows[:MAX_ROWS_SHOWN]:
            print("  " + " | ".join("NULL" if v is None else str(v) for v in row))
        if len(rows) > MAX_ROWS_SHOWN:
            print(f"  ... {len(rows) - MAX_ROWS_SHOWN} more")


def test_sql_lane_graph():
    load_dotenv(ROOT / ".env")
    lane = build_sql_lane()              # one graph, one shared DB connection, reused

    for question in QUESTIONS:
        start = time.time()
        result = lane.invoke({"question": question})   # fresh state per question
        print_result(question, result, time.time() - start)
        assert result.get("status") in {"ok", "failed", "unanswerable"}, "lane ended without a status"


if __name__ == "__main__":
    test_sql_lane_graph()
