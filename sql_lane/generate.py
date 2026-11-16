"""Generate SQL: the only node of the sql_query lane that writes SQL (step2-sql-lane-rules.md, B2).

One LLM call per attempt. Attempt 1 uses the fast model; attempts 2 and 3 (retries) use the
strong model and also see every earlier SQL with the error it caused. The response is a
structured GenerateSqlOutput, so a malformed answer becomes a recorded failure, not a crash.
"""
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, model_validator

from sql_lane.catalog import load_catalog
from sql_lane.rules import AMBIGUITY_DEFAULTS, DATA_LIMITS, GLOBAL_RULES, SQL_RULES
from sql_lane.state import SqlLaneState


class ShapeModel(BaseModel):
    """The result the SQL will return; Validate checks the rows against it."""
    kind: Literal["single_value", "list", "grouped", "time_series"]
    max_rows: int | None = Field(description="Most rows the query can return (its LIMIT), or null if unlimited.")
    order_matters: bool = Field(description="True when the question implies an order.")


class GenerateSqlOutput(BaseModel):
    """What the LLM returns. Fields are written in this order: assumptions before sql."""
    answerable: bool = Field(description="False only if the data cannot answer the question.")
    unanswerable_reason: str | None = Field(description="Why it can't be answered; null if answerable.")
    assumptions: list[str] = Field(description="Every default or interpretation used, in plain words.")
    sql: str | None = Field(description="One read-only Postgres query; null if not answerable.")
    expected_shape: ShapeModel | None = Field(description="Shape of the result; null if not answerable.")

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable:
            if not (self.sql and self.sql.strip()) or self.expected_shape is None:
                raise ValueError("answerable is true, so sql and expected_shape are required")
        elif not self.unanswerable_reason or self.sql or self.expected_shape is not None:
            raise ValueError("answerable is false, so unanswerable_reason is required and sql and expected_shape must be null")
        return self


# ---------------------------------------------------------------- prompt building

def render_data_facts() -> str:
    """The date ranges from the catalog's data_facts section."""
    lines = []
    for f in load_catalog()["data_facts"]:
        lines.append(f"- {f['table']}.{f['column']}: from {f['earliest']} to {f['latest']}. {f['note']}")
    return "\n".join(lines)


def render_schema_context(schema_context: dict[str, dict]) -> str:
    """Turn schema_context (from schema select) into prompt text, one block per table."""
    blocks = []
    for name, t in schema_context.items():
        lines = [f"### {name} ({t['row_count']:,} rows; primary key: {', '.join(t['primary_key'])})",
                 t["description"], "Columns:"]
        for c in t["columns"]:
            line = f"- {c['name']} ({c['type']}): {c['description']}"
            if "values" in c:
                line += f" Values: {', '.join(c['values'])}."
            elif "distinct_values" in c:
                line += f" ({c['distinct_values']:,} distinct values.)"
            if "min" in c:
                line += f" Range: {c['min']} to {c['max']}."
            lines.append(line)
        if t["joins"]:
            lines.append("Joins:")
            for j in t["joins"]:
                kind = "foreign key" if j["type"] == "foreign_key" else "LEFT JOIN only: some rows have no match"
                lines.append(f"- {j['from']} = {j['to']} ({kind})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_system_prompt(schema_context: dict[str, dict]) -> str:
    """Role, global rules, data facts, schema, defaults and limits, SQL rules (spec B2 order)."""
    return f"""You write one read-only PostgreSQL query that answers a question about Olist, a \
Brazilian e-commerce marketplace, or you say the data cannot answer it.

# Global rules (every query must follow them)
{GLOBAL_RULES}

# Data facts
{render_data_facts()}

# Schema (the only tables you may use)
{render_schema_context(schema_context)}

# Ambiguity defaults
{AMBIGUITY_DEFAULTS}

# What the data cannot answer
{DATA_LIMITS}

# How to write the SQL
{SQL_RULES}

# Output
Fill every field. Write assumptions first, then the SQL that follows them."""


def build_user_message(question: str, previous_attempts: list[dict]) -> str:
    """Earlier failed attempts (on a retry), then the question last."""
    parts = []
    if previous_attempts:
        parts.append("Your earlier attempts failed. Fix the problem; do not repeat a failed query.")
        for i, a in enumerate(previous_attempts, 1):
            if a["sql"]:
                parts.append(f"Attempt {i} SQL:\n{a['sql']}\nFailed at {a['stage']}: {a['message']}")
            else:
                parts.append(f"Attempt {i}: your response could not be used: {a['message']}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM call and node

def generate_sql(question: str, schema_context: dict[str, dict],
                 previous_attempts: list[dict], llm: BaseChatModel) -> GenerateSqlOutput:
    """Ask the LLM for one SQL query that answers `question`. One LLM call, no other logic.

    Input:
        question: one complete plain-English question about the Olist data.
        schema_context: from schema select (the node's state["schema_context"]): the
            selected tables with their columns, descriptions, allowed values and joins.
        previous_attempts: [] on the first attempt. On a retry, every earlier attempt as
            {"sql": str, "stage": str, "message": str}: the SQL tried ("" if the response
            couldn't be used), where it failed (generate, review, execute, validate, empty)
            and the exact error.
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o", temperature=0).

    Output: GenerateSqlOutput with
        answerable: False only when the data cannot answer the question.
        unanswerable_reason: why not (null when answerable).
        assumptions: every default or interpretation used, e.g. "Revenue excludes freight",
            "'This year' means 2018, the latest year in the data".
        sql: one PostgreSQL SELECT/WITH query (null when not answerable).
        expected_shape: {kind, max_rows, order_matters} describing the result (null when
            not answerable).

    What a good answer looks like:
        - The SQL answers exactly the question asked, using only the tables in
          schema_context, and follows every global rule (revenue excludes canceled and
          unavailable orders and freight; units = COUNT(*) of order_items; average score
          uses is_latest_for_order; delivery questions filter delivered and NOT
          has_date_issue; never joins two of order_items/order_payments/order_reviews
          without aggregating each per order_id first; ...).
        - Vague words get the documented defaults ("top" = 10, "area" = state, "this year"
          = latest year in the data), and each default appears in assumptions.
        - Filters use the exact listed values (health_beauty, not "Health & Beauty").
        - Computed columns have snake_case aliases with the agreed endings (revenue,
          _count, _pct, score, _days, ...) and money/averages are rounded to 2 decimals.
        - expected_shape matches the SQL (e.g. "top 5" -> list, max_rows 5, order_matters true).
        - Questions needing product/person/page names, profit or margin, forecasts, or
          dates outside the data are answerable = false with a clear reason and no SQL.
        - On a retry, the new SQL fixes the reported error instead of repeating it.

    Raises OutputParserException or ValidationError if the response can't be used; the
    node records that as a generate failure.
    """
    structured = llm.with_structured_output(GenerateSqlOutput, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(build_system_prompt(schema_context)),
                              HumanMessage(build_user_message(question, previous_attempts))])


def make_generate(fast_llm: BaseChatModel, strong_llm: BaseChatModel):
    """Build the Generate SQL node: fast_llm on attempt 1, strong_llm on attempts 2 and 3.

    e.g. make_generate(ChatOpenAI(model="gpt-4o", temperature=0),
                       ChatOpenAI(model="gpt-4o", temperature=0))
    """

    def generate(state: SqlLaneState) -> dict:
        sqls, errors = state.get("sql", []), state.get("errors", [])
        attempt = len(sqls) + 1
        llm = fast_llm if attempt == 1 else strong_llm
        # every earlier attempt failed with exactly one error, so the lists line up
        previous = [{"sql": s, "stage": e["stage"], "message": e["message"]}
                    for s, e in zip(sqls, errors)]
        try:
            out = generate_sql(state["question"], state["schema_context"], previous, llm)
        except (OutputParserException, ValidationError) as exc:
            return {"sql": [""], "errors": [{"stage": "generate", "message": str(exc).strip()}]}

        if not out.answerable:
            # nothing appended to sql: routing checks answerable first and ends the lane
            return {"answerable": False, "unanswerable_reason": out.unanswerable_reason,
                    "assumptions": out.assumptions, "expected_shape": None}
        return {"answerable": True, "unanswerable_reason": None,
                "sql": [out.sql.strip()],
                "expected_shape": out.expected_shape.model_dump(),
                "assumptions": out.assumptions}

    return generate
