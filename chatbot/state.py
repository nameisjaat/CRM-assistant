"""State of the chatbot: the top-level graph above the two lanes.

One invocation = one turn. Short-term memory lives here: with a checkpointer and the same
thread_id, `messages` keeps the whole conversation; `start` clears every per-turn field.
Both lanes run without a checkpointer (fresh each time).

`tasks` and `results` have the same shape on both routes, so one synthesizer serves both:
  sql_query route  tasks = [{"id": "A", "question": standalone_question}], results = {"A": ...}
  analysis route   the analysis planner's tasks and their results
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

HISTORY_MESSAGES = 10              # last 5 turns, shown to the router and the synthesizer


class ChatbotState(TypedDict, total=False):
    # conversation (kept across turns by the checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]

    # this turn
    question: str                  # as the user typed it
    route: Literal["sql_query", "analysis", "out_of_scope"] | None
    standalone_question: str       # the router's rewrite (follow-ups resolved)
    tasks: list[dict]              # [{"id", "question"}]
    results: dict[str, dict]       # task id -> TaskResult (analysis_lane/state.py)
    answer: str
    synth_attempts: int            # 1, or 2 after a grounding retry
    ungrounded_numbers: list[str]  # numbers in the answer not found in any allowed source
    status: Literal["ok", "failed", "unanswerable", "out_of_scope"] | None


def render_history(history: list[AnyMessage]) -> str:
    """The conversation as 'User: ...' / 'Assistant: ...' lines, oldest first."""
    lines = []
    for m in history:
        who = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else None
        if who:
            lines.append(f"{who}: {m.content}")
    return "\n".join(lines)
