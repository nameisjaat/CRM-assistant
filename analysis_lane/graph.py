"""The analysis lane as a LangGraph graph.

    lane = build_analysis_lane()
    out = lane.invoke({"question": "How did revenue and orders change from 2017 to 2018?",
                       "messages": history})            # earlier conversation, optional
    out["tasks"], out["results"]                         # the chatbot's synthesizer writes the answer

START -> start -> planner -(plan problem, once)-> planner
                         -> run_task x N (parallel Send; each runs the sql_query lane) -> END

  start     appends the question to `messages` and clears every per-turn field.
  planner   LLM plan + code check; replans once, then falls back to one task. Reads the
            history in `messages` to turn follow-ups into standalone tasks.
  run_task  one full sql_query lane run per task (the lane is unchanged; it gets only the
            standalone task question, never the chat history).

The lane writes no answer: the chatbot's shared synthesizer (chatbot/synthesize.py) turns
`tasks` + `results` into the answer, for this lane and for the sql_query lane alike.
Short-term memory lives in the chatbot, which passes the history in `messages`.

The sql_query lane is compiled once and shared by every task. Its database connection
lets one query in at a time; the LLM calls of parallel tasks overlap.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from analysis_lane.planner import make_planner
from analysis_lane.state import AnalysisState, Task
from sql_lane.execute import ConnectionFailure
from sql_lane.graph import build_sql_lane

PLANNER_MODEL = "gpt-4o-mini"      # project owner's decision


# ---------------------------------------------------------------- nodes

def start(state: AnalysisState) -> dict:
    """New run: add the question after the history, clear per-run fields."""
    return {"messages": [HumanMessage(content=state["question"])],
            "tasks": [], "results": None,                # None clears (merge_results)
            "plan_attempts": 0, "plan_error": None}


def make_run_task(sql_lane):
    """Node that runs one task through the sql_query lane.

    It receives the Send payload ({"id", "question"}, typed as Task), not the graph state,
    and returns {"results": {id: result}}, which merge_results adds to the state.
    """

    def run_task(task: Task) -> dict:
        return {"results": {task["id"]: run_sql_task(sql_lane, task["question"])}}

    return run_task


def run_sql_task(sql_lane, question: str) -> dict:
    """Run one question through the sql_query lane; return a TaskResult.

    Shared with the chatbot's sql_query route, so both routes hand the synthesizer
    results of the same shape. A ConnectionFailure (database down) is raised; any other
    error becomes a failed result so one task can't sink the rest.
    """
    try:
        out = sql_lane.invoke({"question": question})            # fresh lane state
        errors = out.get("errors", [])
        return {"question": question, "status": out.get("status", "failed"),
                "columns": out.get("columns", []), "rows": out.get("rows", []),
                "truncated": out.get("truncated", False),
                "assumptions": out.get("assumptions", []),
                "unanswerable_reason": out.get("unanswerable_reason"),
                "sql": (out.get("sql") or [None])[-1] or None,
                "error": errors[-1]["message"] if errors and out.get("status") != "ok" else None}
    except ConnectionFailure:
        raise
    except Exception as e:
        return {"question": question, "status": "failed", "columns": [], "rows": [],
                "truncated": False, "assumptions": [], "unanswerable_reason": None,
                "sql": None, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- routing

def after_planner(state: AnalysisState):
    """Replan once if the plan had a problem; otherwise one Send per task (parallel)."""
    if state.get("plan_error"):
        return "planner"
    return [Send("run_task", {"id": t["id"], "question": t["question"]}) for t in state["tasks"]]


# ---------------------------------------------------------------- building

def build_analysis_lane(planner_llm: BaseChatModel | None = None, sql_lane=None,
                        checkpointer=None):
    """The analysis lane with the default planner model (gpt-4o-mini) and sql_query lane.

    checkpointer: None by default (the chatbot keeps the memory and passes the history
    in `messages`). Needs OPENAI_API_KEY and DATABASE_URL in the environment.
    """
    planner_llm = planner_llm or ChatOpenAI(model=PLANNER_MODEL, temperature=0)
    sql_lane = sql_lane or build_sql_lane()

    g = StateGraph(AnalysisState)
    g.add_node("start", start)
    g.add_node("planner", make_planner(planner_llm))
    g.add_node("run_task", make_run_task(sql_lane))

    g.add_edge(START, "start")
    g.add_edge("start", "planner")
    g.add_conditional_edges("planner", after_planner, ["planner", "run_task"])
    g.add_edge("run_task", END)                       # the run ends when every task is done
    return g.compile(checkpointer=checkpointer)
