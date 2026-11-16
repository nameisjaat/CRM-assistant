"""State shared by every node of the sql_query lane (step2-sql-lane-rules.md, Part E).

Each node is a function (state) -> dict that returns only the fields it changes.
`sql` and `errors` use the operator.add reducer: a node returns just the new item,
e.g. {"sql": [new_sql]}, and LangGraph appends it to the list. So:

- attempt number          = len(sql)   (every attempt adds exactly one SQL; "" if
                                        Generate's response couldn't be parsed)
- current attempt failed  = len(errors) == len(sql)   (at most one error per attempt)

Every question must start with a fresh state (a new invocation), or the lists would
carry over from an earlier question.
"""
import operator
from typing import Annotated, Literal, TypedDict


class ExpectedShape(TypedDict):
    """What Generate SQL says the result will look like; Validate checks rows against it."""
    kind: Literal["single_value", "list", "grouped", "time_series"]
    max_rows: int | None          # None when the question sets no limit (e.g. "revenue by state")
    order_matters: bool           # True when the question implies an order ("top", "by month")


class LaneError(TypedDict):
    """One failed attempt: which step failed and the exact message fed back to Generate SQL."""
    stage: Literal["generate", "review", "execute", "validate", "empty"]
    message: str


class SqlLaneState(TypedDict, total=False):
    # input
    question: str                                 # the router's standalone question

    # schema select
    tables: list[str]                             # selected table names
    schema_context: dict[str, dict]               # per table, built from the schema catalog

    # generate
    answerable: bool
    unanswerable_reason: str | None
    sql: Annotated[list[str], operator.add]       # every attempt; current SQL = sql[-1]
    expected_shape: ExpectedShape | None          # None when answerable is False
    assumptions: list[str]

    # execute
    columns: list[str]
    rows: list[list]                              # JSON-safe values only
    truncated: bool                               # more than 1,000 rows existed

    # generate / review / execute / validate failures
    errors: Annotated[list[LaneError], operator.add]
    empty_retry_done: bool                        # the one retry for an empty result has been used

    # output
    status: Literal["ok", "failed", "unanswerable"]
