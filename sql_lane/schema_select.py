"""Schema select: the first node of the sql_query lane.

An LLM picks which tables the question needs; plain code then completes the list
(join tables, orders, the 6-table cap) and builds `schema_context` from the catalog.
Writes `tables` and `schema_context` to the state. Runs once per question; retries
reuse its result.
"""
from functools import lru_cache
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from sql_lane.catalog import (GEOLOCATION, build_schema_context, connect_tables,
                              load_catalog, table_names)
from sql_lane.state import SqlLaneState

MAX_TABLES = 6

# Used when the LLM returns no tables or an unusable response (spec B1 rule 5).
DEFAULT_TABLES = ["orders", "order_items", "customers", "products", "sellers"]

# One-to-many tables hanging off orders; when any is selected, orders is added because
# the status filters (e.g. excluding canceled orders from revenue) live there.
ORDER_CHILDREN = {"order_items", "order_payments", "order_reviews"}

# Only the real table names can be returned: OpenAI's structured output enforces this list.
TableName = Literal[tuple(table_names())]


class TableSelection(BaseModel):
    """What the LLM returns. `reason` comes first so the model explains before choosing."""
    reason: str = Field(description="One or two sentences: why these tables are needed.")
    tables: list[TableName] = Field(
        description="Every table needed to answer the question, most important first.")


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """Instructions plus a description of every table (same text on every call)."""
    blocks = []
    for table in load_catalog()["tables"]:
        lines = [f"## {table['name']}", table["description"], "Columns:"]
        for col in table["columns"]:
            line = f"- {col['name']}: {col['description']}"
            if "values" in col:
                line += f" Values: {', '.join(col['values'])}."
            lines.append(line)
        blocks.append("\n".join(lines))
    tables_text = "\n\n".join(blocks)

    return f"""You choose which tables of a Postgres database are needed to answer a question \
about Olist, a Brazilian e-commerce marketplace.

Rules:
- Pick every table needed: for the values the question asks for, its filters, its grouping, \
and the joins between them.
- When unsure whether a table is needed, include it. A missing table makes the question \
impossible to answer; an extra table costs little.
- List the tables from most to least important.
- Do not judge whether the question can be answered. Always pick the closest tables.
- Explain your choice briefly in `reason`.

The tables:

{tables_text}"""


def select_tables(question: str, llm: BaseChatModel) -> TableSelection:
    """Ask the LLM which tables are needed to answer `question`. One LLM call, no other logic.

    Input:
        question: one complete plain-English question about the Olist data, e.g.
            "Which 10 sellers had the most revenue in 2018?"
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0).

    Output: TableSelection with
        reason: a short explanation of the choice.
        tables: table names, most important first. Only the 10 real tables can appear:
            closed_deals, customers, geolocation, leads, order_items, order_payments,
            order_reviews, orders, products, sellers.

    What a good answer looks like:
        - Every table the SQL needs is present: the tables holding the values asked for
          (revenue = order_items, review score = order_reviews, conversion = leads and
          closed_deals, ...), the tables used to filter or group (e.g. products for
          "by category", customers for "customer home state"), and the tables in between
          that join them (e.g. orders between customers and order_items).
        - Missing a needed table is a serious error. An extra table is only a minor cost.
        - It never refuses: even for a question the data can't answer (product names,
          forecasts), it still returns the closest tables. Answerability is decided
          later by Generate SQL.

    Not done here (done by the node afterwards, in code): adding missing join tables,
    adding orders, the 6-table cap and the fallback. Raises if the response can't be
    parsed; the node catches that.
    """
    structured = llm.with_structured_output(TableSelection, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(system_prompt()), HumanMessage(question)])


def finalize_tables(picked: list[str]) -> list[str]:
    """Apply the code rules to the LLM's ranked picks; returns tables in catalog order.

    1. Add the in-between tables needed to join the picks (connect_tables).
    2. Add orders when order_items, order_payments or order_reviews is present.
    3. While more than MAX_TABLES remain: drop geolocation if picked, otherwise drop the
       lowest-ranked pick, and redo steps 1-2 (so join tables added only for the
       dropped pick disappear too).
    Empty or unknown picks fall back to DEFAULT_TABLES.
    """
    known = set(table_names())
    picks = [t for t in dict.fromkeys(picked) if t in known]     # dedupe, keep rank
    if not picks:
        picks = list(DEFAULT_TABLES)

    while True:
        tables = connect_tables(picks)
        if ORDER_CHILDREN & set(tables) and "orders" not in tables:
            tables.append("orders")       # orders joins each child directly
        if len(tables) <= MAX_TABLES or len(picks) == 1:
            break
        if GEOLOCATION in picks:
            picks.remove(GEOLOCATION)
        else:
            picks.pop()                   # lowest-ranked pick

    order = table_names()
    return sorted(tables, key=order.index)


def make_schema_select(llm: BaseChatModel):
    """Build the schema select node around `llm` (e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0))."""

    def schema_select(state: SqlLaneState) -> dict:
        try:
            picked = select_tables(state["question"], llm).tables
        except (OutputParserException, ValidationError):
            picked = []                   # unusable response -> DEFAULT_TABLES
        tables = finalize_tables(picked)
        return {"tables": tables, "schema_context": build_schema_context(tables)}

    return schema_select
