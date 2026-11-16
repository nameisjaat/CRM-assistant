"""The chatbot: router on top, the two lanes below, one shared synthesizer.

    bot = build_chatbot(checkpointer=InMemorySaver())        # memory needs a checkpointer
    config = {"configurable": {"thread_id": "chat-1"}}
    bot.invoke({"question": "Revenue by state in 2018"}, config)
    bot.invoke({"question": "What about 2017?"}, config)      # follow-up

                         ┌ sql_query ──── run_sql ───────┐
START -> start -> router ┤                               ├-> synthesize -(once)-> synthesize -> END
                         ├ analysis ───── run_analysis ──┘
                         └ out_of_scope ─ out_of_scope ───────────────────────────────────────> END

  start         appends the question to `messages`, clears every per-turn field.
  router        LLM: route + standalone question (follow-ups resolved from history).
  run_sql       the sql_query lane on the standalone question, stored as task "A".
  run_analysis  the analysis lane (planner + parallel tasks) on the standalone question,
                with the earlier conversation passed in `messages`.
  out_of_scope  a fixed reply; no LLM.
  synthesize    the one answer writer for both lanes (chatbot/synthesize.py); appends the
                answer to `messages` with its `grounded` mark.

Short-term memory lives only here (the checkpointer). Both lanes run without one. One
sql_query lane is compiled and shared by run_sql and every analysis task.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from analysis_lane.graph import build_analysis_lane, run_sql_task
from chatbot.router import make_router
from chatbot.state import ChatbotState
from chatbot.synthesize import make_synthesize
from sql_lane.graph import build_sql_lane

ROUTER_MODEL = "gpt-4o-mini"       # project owner's decision
SYNTH_MODEL = "gpt-4o-mini"

OUT_OF_SCOPE_REPLY = (
    "I can only answer questions about the Olist e-commerce data: orders and revenue, "
    "products and categories, sellers, customers, reviews, payments, deliveries and the "
    "marketing funnel (leads and closed deals). For example: \"Which 5 categories had the "
    "most revenue in 2018?\"")


# ---------------------------------------------------------------- nodes

def start(state: ChatbotState) -> dict:
    """New turn: remember the question, clear everything left from the previous turn."""
    return {"messages": [HumanMessage(content=state["question"])],
            "route": None, "standalone_question": "", "tasks": [], "results": {},
            "answer": "", "synth_attempts": 0, "ungrounded_numbers": [], "status": None}


def make_run_sql(sql_lane):
    def run_sql(state: ChatbotState) -> dict:
        question = state["standalone_question"]
        return {"tasks": [{"id": "A", "question": question}],
                "results": {"A": run_sql_task(sql_lane, question)}}
    return run_sql


def make_run_analysis(analysis_lane):
    def run_analysis(state: ChatbotState) -> dict:
        history = state.get("messages", [])[:-1]          # before this turn's question
        out = analysis_lane.invoke({"question": state["standalone_question"], "messages": history})
        return {"tasks": out.get("tasks", []), "results": out.get("results", {})}
    return run_analysis


def out_of_scope(state: ChatbotState) -> dict:
    return {"answer": OUT_OF_SCOPE_REPLY, "status": "out_of_scope",
            "messages": [AIMessage(content=OUT_OF_SCOPE_REPLY, additional_kwargs={"grounded": True})]}


# ---------------------------------------------------------------- routing

def after_router(state: ChatbotState) -> str:
    return {"sql_query": "run_sql", "analysis": "run_analysis"}.get(state.get("route"), "out_of_scope")


def after_synthesize(state: ChatbotState) -> str:
    """The final synthesize call sets status; without it, synthesize runs once more."""
    return END if state.get("status") else "synthesize"


# ---------------------------------------------------------------- building

def build_chatbot(router_llm: BaseChatModel | None = None, synth_llm: BaseChatModel | None = None,
                  sql_lane=None, analysis_lane=None, checkpointer=None):
    """The chatbot with default models (gpt-4o-mini router and synthesizer) and lanes.

    checkpointer: e.g. InMemorySaver() for short-term memory across turns of the same
    thread_id; None makes every invoke independent. Needs OPENAI_API_KEY and DATABASE_URL
    in the environment (load the project's .env first).
    """
    router_llm = router_llm or ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    synth_llm = synth_llm or ChatOpenAI(model=SYNTH_MODEL, temperature=0)
    sql_lane = sql_lane or build_sql_lane()
    analysis_lane = analysis_lane or build_analysis_lane(sql_lane=sql_lane)

    g = StateGraph(ChatbotState)
    g.add_node("start", start)
    g.add_node("router", make_router(router_llm))
    g.add_node("run_sql", make_run_sql(sql_lane))
    g.add_node("run_analysis", make_run_analysis(analysis_lane))
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("synthesize", make_synthesize(synth_llm))

    g.add_edge(START, "start")
    g.add_edge("start", "router")
    g.add_conditional_edges("router", after_router, ["run_sql", "run_analysis", "out_of_scope"])
    g.add_edge("run_sql", "synthesize")
    g.add_edge("run_analysis", "synthesize")
    g.add_edge("out_of_scope", END)
    g.add_conditional_edges("synthesize", after_synthesize, ["synthesize", END])
    return g.compile(checkpointer=checkpointer)
