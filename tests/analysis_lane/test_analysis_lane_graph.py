"""Smoke test: run the analysis lane on two questions and print the plan and task results.

    .venv\\Scripts\\python.exe tests\\analysis_lane\\test_analysis_lane_graph.py

The analysis lane no longer writes an answer (the chatbot's shared synthesizer does) and
keeps no memory (the chatbot does), so this prints only what the lane produces: the
planner's standalone tasks and each task's sql_query lane result. The second question is
run with a hand-written history to show the planner resolving a follow-up.
For the full experience (router, answers, memory) run tests\\chatbot\\test_chatbot_graph.py.

Calls OpenAI (gpt-4o-mini for planner, schema select and review; gpt-4o for generate
inside each task) and queries the Neon database (read-only).
Reads OPENAI_API_KEY and DATABASE_URL from the project's .env.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # so `analysis_lane` and `sql_lane` import

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from analysis_lane.graph import build_analysis_lane

RUNS = [
    # (question, history passed in `messages`)
    ("Do late deliveries get lower review scores than on-time ones, and which 3 delivery "
     "states have the highest share of late orders?", []),
    ("What about the average review score in those years?",
     [HumanMessage("How did revenue and the number of orders change from 2017 to 2018?"),
      AIMessage("Revenue and orders both grew from 2017 to 2018.")]),
]
ROWS_SHOWN = 8


def print_run(n: int, question: str, state: dict, seconds: float) -> None:
    line = "=" * 90
    print(f"\n{line}\nRun {n}: {question}\n{line}")
    print(f"({seconds:.1f}s; plan attempts {state.get('plan_attempts')})")

    print("\nPlan (standalone tasks sent to the sql_query lane):")
    for t in state.get("tasks", []):
        print(f"  {t['id']}. {t['question']}")

    for t in state.get("tasks", []):
        r = state.get("results", {}).get(t["id"], {})
        print(f"\n--- Task {t['id']}: {r.get('status')} ---")
        if r.get("unanswerable_reason"):
            print(f"Unanswerable: {r['unanswerable_reason']}")
        if r.get("error"):
            print(f"Error: {r['error']}")
        if r.get("sql"):
            print(r["sql"])
        if r.get("columns"):
            print("  " + " | ".join(r["columns"]))
            for row in r.get("rows", [])[:ROWS_SHOWN]:
                print("  " + " | ".join("NULL" if v is None else str(v) for v in row))
            if len(r.get("rows", [])) > ROWS_SHOWN:
                print(f"  ... {len(r['rows']) - ROWS_SHOWN} more")


def test_analysis_lane_graph():
    load_dotenv(ROOT / ".env")
    lane = build_analysis_lane()

    for n, (question, history) in enumerate(RUNS, 1):
        start = time.time()
        state = lane.invoke({"question": question, "messages": history})
        print_run(n, question, state, time.time() - start)
        assert state.get("tasks"), "the planner produced no tasks"
        assert set(state.get("results", {})) == {t["id"] for t in state["tasks"]}, \
            "every task should have a result"


if __name__ == "__main__":
    test_analysis_lane_graph()
