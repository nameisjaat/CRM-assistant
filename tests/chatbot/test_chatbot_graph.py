"""Smoke test: one conversation with the chatbot (router, both lanes, synthesizer, memory).

    .venv\\Scripts\\python.exe tests\\chatbot\\test_chatbot_graph.py

Uses InMemorySaver and one thread_id, so the turns share short-term memory:
  1. a sql_query question
  2. a follow-up to it (the router must make it standalone from the history)
  3. an analysis question
  4. an out-of-scope message (fixed reply, no lane)

Calls OpenAI (gpt-4o-mini for router, synthesizer, planner, schema select and review;
gpt-4o for generate) and queries the Neon database (read-only).
Reads OPENAI_API_KEY and DATABASE_URL from the project's .env.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # so `chatbot`, `analysis_lane` and `sql_lane` import

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from chatbot.graph import build_chatbot

THREAD_ID = "chatbot-smoke-1"
CONVERSATION = [
    "Which 5 product categories had the most revenue in 2018?",                 # sql_query
    "What about in 2017?",                                                      # follow-up
    "How did revenue, number of orders and average review score change from 2017 to 2018?",  # analysis
    "Hi! Can you tell me a joke?",                                              # out_of_scope
]
ROWS_SHOWN = 8


def print_turn(n: int, question: str, state: dict, seconds: float) -> None:
    line = "=" * 90
    print(f"\n{line}\nTurn {n}: {question}\n{line}")
    print(f"Route:       {state.get('route')}")
    print(f"Standalone:  {state.get('standalone_question')}")
    print(f"Status:      {state.get('status')}   ({seconds:.1f}s, synthesize attempts "
          f"{state.get('synth_attempts')})")

    for t in state.get("tasks", []):
        r = state.get("results", {}).get(t["id"], {})
        print(f"\n--- Task {t['id']}: {t['question']}  [{r.get('status')}]")
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

    print(f"\nAnswer:\n{state.get('answer')}")
    if state.get("ungrounded_numbers"):
        print(f"\nUngrounded numbers (not found in any result): {state['ungrounded_numbers']}")
    print(f"\nMessages in memory: {len(state.get('messages', []))}")


def test_chatbot_graph():
    load_dotenv(ROOT / ".env")
    bot = build_chatbot(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": THREAD_ID}}

    for n, question in enumerate(CONVERSATION, 1):
        start = time.time()
        state = bot.invoke({"question": question}, config)
        print_turn(n, question, state, time.time() - start)
        assert state.get("status") in {"ok", "failed", "unanswerable", "out_of_scope"}, \
            "turn ended without a status"
        assert len(state.get("messages", [])) == 2 * n, "memory should hold every question and answer"


if __name__ == "__main__":
    test_chatbot_graph()
