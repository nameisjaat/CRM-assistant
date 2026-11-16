"""Planner: splits the user's question into independent sub-questions for the sql_query lane.

One LLM call (gpt-4o-mini). Plain code then checks the plan (check_plan). If the plan has
a problem (more than MAX_TASKS tasks, no usable task), the planner is asked ONCE more with
its previous plan and the exact problem, so the LLM rebuilds the plan instead of code
cutting it blindly. If the second plan also fails, the lane falls back to one task: the
original question.

The planner sees the last HISTORY_MESSAGES messages of the conversation, so it can turn a
follow-up ("what about 2017?") into standalone questions.
"""
from functools import lru_cache

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from analysis_lane.state import AnalysisState
from sql_lane.catalog import load_catalog
from sql_lane.generate import render_data_facts

MAX_TASKS = 6
HISTORY_MESSAGES = 10              # last 5 turns (question + answer each) shown to the planner
TASK_IDS = "ABCDEFGHIJ"


class PlannedTask(BaseModel):
    question: str = Field(description="One complete, standalone question that one SQL query "
                                      "can answer, with every name, period and filter spelled out.")


class Plan(BaseModel):
    """What the LLM returns. `reasoning` comes first so it thinks before splitting."""
    reasoning: str = Field(description="Briefly: what the user wants and how you split it.")
    tasks: list[PlannedTask] = Field(description=f"1 to {MAX_TASKS} independent sub-questions.")


# ---------------------------------------------------------------- prompt

@lru_cache(maxsize=1)
def _tables_text() -> str:
    """Each table with its description and column names (no values: the SQL lane has them)."""
    lines = []
    for t in load_catalog()["tables"]:
        cols = ", ".join(c["name"] for c in t["columns"])
        lines.append(f"- {t['name']}: {t['description']} Columns: {cols}.")
    return "\n".join(lines)


def build_planner_system_prompt() -> str:
    return f"""You plan how to answer a question about Olist, a Brazilian e-commerce \
marketplace, by splitting it into sub-questions. Each sub-question is answered by a separate \
SQL assistant that writes ONE read-only SQL query; you never write SQL.

# Rules for the tasks
1. Each task is ONE complete, standalone question that one SQL query can answer. Spell out \
every name, period, filter and number: the SQL assistant does not see the conversation or \
the other tasks.
2. Tasks are independent: no task may use another task's result. If one step needs another \
step's result ("review scores for the 5 categories with the most late deliveries"), write \
it as ONE task; the SQL assistant handles that in a single query.
3. Together the tasks must cover every part of the user's question. No duplicates.
4. Use 1 to {MAX_TASKS} tasks, as few as the question needs. A simple question is 1 task.
5. For a comparison ("2017 vs 2018"), prefer one task that returns both sides (e.g. grouped \
by year) over two tasks, so the numbers come from the same query.
6. For an explanation ("why did X drop?"), ask for the numbers that could show the reason \
(the trend itself, and the same measure broken down by state, category, payment method, \
delivery delay, ...), not for the reason itself.
7. Do not judge whether the data can answer a task; the SQL assistant decides. Keep the \
user's wording for vague terms ("this year", "top", "area"): the SQL assistant applies the \
standard defaults.
8. If the question is a follow-up ("what about 2017?", "and for them?"), use the \
conversation so far to make every task standalone.

# Data facts
{render_data_facts()}

# The tables (for orientation only)
{_tables_text()}"""


def _render_history(history: list[AnyMessage]) -> str:
    lines = []
    for m in history:
        who = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else None
        if who:
            lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def build_planner_user_message(question: str, history: list[AnyMessage],
                               plan_error: str | None) -> str:
    parts = []
    if history:
        parts.append(f"Conversation so far:\n{_render_history(history)}")
    if plan_error:
        parts.append(f"Your previous plan could not be used.\n{plan_error}\nMake a new plan "
                     f"that fixes this and still covers every part of the question.")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM call and plan check

def plan_tasks(question: str, history: list[AnyMessage], plan_error: str | None,
               llm: BaseChatModel) -> Plan:
    """Ask the LLM to split `question` into independent sub-questions. One LLM call.

    Input:
        question: the user's question for this turn (may be a follow-up).
        history: the earlier conversation, oldest first: HumanMessage (user questions) and
            AIMessage (assistant answers); the node passes the last HISTORY_MESSAGES.
        plan_error: None on the first call. On a replan: the previous plan and its
            problem, e.g. "Previous plan: 1. ... 8. ...\\nProblem: 8 tasks; at most 6."
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0).

    Output: Plan with
        reasoning: a short explanation of the split.
        tasks: 1 to 6 PlannedTask(question=...).

    What a good plan looks like:
        - Every task is one standalone question that a single SQL query can answer, with
          names, periods and filters spelled out (no "them", "that year", "the above").
        - Tasks are independent; a step that needs another step's result is merged into
          one task ("average review score of the 5 categories with the most late
          deliveries").
        - Together the tasks cover every part of the question, with no duplicates and no
          extra questions the user did not ask.
        - Comparisons come back from one task where possible ("revenue by year for 2017
          and 2018"); "why" questions ask for the numbers that could explain it.
        - A follow-up ("what about 2017?") is resolved from the history into standalone
          tasks.
        - On a replan, the new plan fixes the stated problem (e.g. merges 8 tasks into 6)
          without dropping any part of the question.

    Raises OutputParserException / ValidationError if the response can't be used; the
    node counts that as a failed plan.
    """
    structured = llm.with_structured_output(Plan, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(build_planner_system_prompt()),
                              HumanMessage(build_planner_user_message(question, history, plan_error))])


def _normalise(q: str) -> str:
    return " ".join(q.lower().split()).rstrip("?. ")


def check_plan(questions: list[str]) -> tuple[list[str], str | None]:
    """Plain code. Returns (clean questions, problem or None).

    Fixed here without loss: surrounding spaces, empty questions and exact duplicates
    (same words, ignoring case and spacing) are removed. Not fixed here (sent back to the
    planner): no questions left, or more than MAX_TASKS.
    """
    clean, seen = [], set()
    for q in questions:
        q = (q or "").strip()
        key = _normalise(q)
        if key and key not in seen:
            seen.add(key)
            clean.append(q)
    if not clean:
        return clean, "Problem: the plan has no usable task."
    if len(clean) > MAX_TASKS:
        listed = "\n".join(f"{i}. {q}" for i, q in enumerate(clean, 1))
        return clean, (f"Previous plan:\n{listed}\nProblem: {len(clean)} tasks; the limit is "
                       f"{MAX_TASKS}. Merge related tasks into fewer questions (for example one "
                       f"grouped question instead of one per year or per state).")
    return clean, None


def _tasks(questions: list[str]) -> list[dict]:
    return [{"id": TASK_IDS[i], "question": q} for i, q in enumerate(questions)]


# ---------------------------------------------------------------- node

def make_planner(llm: BaseChatModel):
    """Build the planner node around `llm` (e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0))."""

    def planner(state: AnalysisState) -> dict:
        question = state["question"]
        attempt = state.get("plan_attempts", 0) + 1
        # the current question is the last message (added by start); history is before it
        history = state.get("messages", [])[:-1][-HISTORY_MESSAGES:]

        try:
            plan = plan_tasks(question, history, state.get("plan_error"), llm)
            questions, problem = check_plan([t.question for t in plan.tasks])
        except (OutputParserException, ValidationError):
            questions, problem = [], "Problem: your response could not be used. Return a valid plan."

        if problem is None:
            return {"tasks": _tasks(questions), "plan_attempts": attempt, "plan_error": None}
        if attempt == 1:
            return {"plan_attempts": attempt, "plan_error": problem}      # replan once
        # second failure: answer the original question as a single task
        return {"tasks": _tasks([question]), "plan_attempts": attempt, "plan_error": None}

    return planner
