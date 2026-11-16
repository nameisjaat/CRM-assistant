"""State of the analysis lane.

One invocation = one question. The caller (the chatbot) passes the earlier conversation in
`messages`; `start` appends the question and clears every per-run field. The lane ends
with `tasks` and `results`; it writes no answer (the chatbot's synthesizer does).

Reducers:
  messages  add_messages: start appends the question after the history.
  results   merge_results: parallel tasks each add {task_id: result} without overwriting
            each other; returning None clears it (used by start).
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_results(current: dict | None, update: dict | None) -> dict:
    """Add parallel task results together; None clears (a new turn starts)."""
    if update is None:
        return {}
    return {**(current or {}), **update}


class Task(TypedDict):
    id: str                       # "A", "B", ... (assigned in code, in plan order)
    question: str                 # one standalone question for the sql_query lane


class TaskResult(TypedDict):
    question: str
    status: Literal["ok", "failed", "unanswerable"]
    columns: list[str]
    rows: list[list]              # JSON-safe, at most 1,000 (sql_query lane cap)
    truncated: bool
    assumptions: list[str]
    unanswerable_reason: str | None
    sql: str | None               # the final SQL the lane ran, if any
    error: str | None             # last error message, if the task failed


class AnalysisState(TypedDict, total=False):
    # conversation (kept across turns)
    messages: Annotated[list[AnyMessage], add_messages]

    # this turn
    question: str
    tasks: list[Task]
    plan_attempts: int            # 1, or 2 after a replan
    plan_error: str | None        # the plan's problem, fed back to the planner (None = ok)
    results: Annotated[dict[str, TaskResult], merge_results]
