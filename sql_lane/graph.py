"""The sql_query lane as a LangGraph graph (step2-sql-lane-rules.md, Part E).

    lane = build_sql_lane()
    result = lane.invoke({"question": "Which 5 sellers had the most revenue in 2018?"})
    result["status"], result["columns"], result["rows"], result["sql"][-1]

START -> schema_select -> generate -> review -> execute -> validate -> END
A failure at generate, review, execute or validate goes back to generate (at most 3
attempts in all). Three small end nodes set the final status:
  give_up       status "failed"        (attempts used up, or the same failing SQL again)
  unanswerable  status "unanswerable"  (Generate SQL said the data can't answer it)
  accept_empty  status "ok", 0 rows    (the same SQL again after an empty result)

Every invoke starts from a fresh state, so sql and errors never carry over between
questions. A ConnectionFailure from Execute is not caught: it comes out of invoke.
No checkpointer yet (it belongs to the outer graph; the state is already JSON-safe).
"""
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from sql_lane.execute import SharedConnection, make_execute
from sql_lane.generate import make_generate
from sql_lane.review import make_review
from sql_lane.schema_select import make_schema_select
from sql_lane.state import SqlLaneState
from sql_lane.validate import MAX_ATTEMPTS, validate

# Models are chosen in code (project owner's decision). Temperature 0 everywhere.
SCHEMA_MODEL = "gpt-4o-mini"
FAST_MODEL = "gpt-4o"             # generate, attempt 1 (project owner: gpt-4o on all attempts)
STRONG_MODEL = "gpt-4o"           # generate, attempts 2 and 3
REVIEW_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------- end nodes

def give_up(state: SqlLaneState) -> dict:
    """No usable answer: failed, and no rows from a failed attempt reach the Writer."""
    return {"status": "failed", "columns": [], "rows": [], "truncated": False}


def unanswerable(state: SqlLaneState) -> dict:
    """Generate SQL said the data can't answer it; clear rows an earlier attempt left."""
    return {"status": "unanswerable", "columns": [], "rows": [], "truncated": False}

def accept_empty(state: SqlLaneState) -> dict:
    """The same SQL again after an empty result: accept "no matching data" (status ok).

    If the most recent result in the state is that empty result, keep it (its column
    names help the Writer say "no orders matched"). If a later attempt ran different SQL
    in between, its rows are not this query's result, so they are cleared.
    """
    if state["errors"][-1]["stage"] == "empty":
        return {"status": "ok"}
    return {"status": "ok", "columns": [], "rows": [], "truncated": False}


# ---------------------------------------------------------------- routing

def attempt_failed(state: SqlLaneState) -> bool:
    """Each attempt adds one SQL and at most one error, so equal lengths = it failed."""
    return len(state.get("errors", [])) == len(state.get("sql", []))


def retry_or_give_up(state: SqlLaneState) -> str:
    return "give_up" if len(state.get("sql", [])) >= MAX_ATTEMPTS else "generate"


def after_generate(state: SqlLaneState) -> str:
    # 1. answerable first: an unanswerable reply appends no SQL, so the failure count
    #    below would be wrong on a retry (project owner's decision)
    if state.get("answerable") is False:
        return "unanswerable"
    # 2. the response couldn't be used
    if attempt_failed(state):
        return retry_or_give_up(state)
    # 3. the same SQL as an earlier attempt would end the same way
    sqls, errors = state["sql"], state.get("errors", [])
    for earlier_sql, earlier_error in zip(sqls[:-1], errors):
        if earlier_sql and earlier_sql == sqls[-1]:
            return "accept_empty" if earlier_error["stage"] == "empty" else "give_up"
    return "review"


def after_review(state: SqlLaneState) -> str:
    return retry_or_give_up(state) if attempt_failed(state) else "execute"


def after_execute(state: SqlLaneState) -> str:
    return retry_or_give_up(state) if attempt_failed(state) else "validate"


def after_validate(state: SqlLaneState) -> str:
    return retry_or_give_up(state) if attempt_failed(state) else END


# ---------------------------------------------------------------- building

def build_graph(schema_select, generate, review, execute, validate_node=validate):
    """Wire the given node functions into the compiled lane graph.

    Takes the nodes themselves, so the routing can also be run with stand-in nodes.
    """
    g = StateGraph(SqlLaneState)
    g.add_node("schema_select", schema_select)
    g.add_node("generate", generate)
    g.add_node("review", review)
    g.add_node("execute", execute)
    g.add_node("validate", validate_node)
    g.add_node("give_up", give_up)
    g.add_node("unanswerable", unanswerable)
    g.add_node("accept_empty", accept_empty)

    g.add_edge(START, "schema_select")
    g.add_edge("schema_select", "generate")
    g.add_conditional_edges("generate", after_generate,
                            ["unanswerable", "generate", "give_up", "accept_empty", "review"])
    g.add_conditional_edges("review", after_review, ["generate", "give_up", "execute"])
    g.add_conditional_edges("execute", after_execute, ["generate", "give_up", "validate"])
    g.add_conditional_edges("validate", after_validate, ["generate", "give_up", END])
    for end_node in ("give_up", "unanswerable", "accept_empty"):
        g.add_edge(end_node, END)
    return g.compile()


def build_sql_lane(schema_llm: BaseChatModel | None = None,
                   fast_llm: BaseChatModel | None = None,
                   strong_llm: BaseChatModel | None = None,
                   review_llm: BaseChatModel | None = None,
                   connection: SharedConnection | None = None):
    """The full lane with the default models and one shared database connection.

    Any model or the connection can be passed in instead. Needs OPENAI_API_KEY and
    DATABASE_URL in the environment (load the project's .env first).
    """
    def model(name: str) -> BaseChatModel:
        return ChatOpenAI(model=name, temperature=0)

    return build_graph(
        schema_select=make_schema_select(schema_llm or model(SCHEMA_MODEL)),
        generate=make_generate(fast_llm or model(FAST_MODEL), strong_llm or model(STRONG_MODEL)),
        review=make_review(review_llm or model(REVIEW_MODEL)),
        execute=make_execute(connection or SharedConnection()),
    )
