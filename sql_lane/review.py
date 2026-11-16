"""Review node: checks the SQL from Generate before it runs (step2-sql-lane-rules.md, B3).

Two layers, in order; the first failure goes back to Generate SQL as feedback:
  1. review_code.code_review  every check code can do on the SQL alone (safety + data rules).
  2. review_llm.llm_review    only what depends on the question, for SQL that passed layer 1.
Both record errors with stage "review"; LLM messages start with "LLM review:".
Review never rewrites the SQL. On success it writes nothing to the state.
"""
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import LengthFinishReasonError
from pydantic import ValidationError

from sql_lane.review_code import code_review
from sql_lane.review_llm import llm_review, review_message
from sql_lane.state import SqlLaneState

# The reviewer's own failure is not the SQL's fault: the SQL passed every code check.
REVIEWER_FAILURES = (OutputParserException, ValidationError, OpenAIRefusalError,
                     LengthFinishReasonError)


def make_review(llm: BaseChatModel):
    """Build the Review node around the reviewer LLM, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0)."""

    def review(state: SqlLaneState) -> dict:
        sql = state["sql"][-1]

        # layer 1: code
        if error := code_review(sql):
            return {"errors": [{"stage": "review", "message": error}]}

        # layer 2: LLM (an unusable reviewer answer approves)
        try:
            verdict = llm_review(state["question"], sql, state.get("assumptions", []),
                                 state["schema_context"], llm)
        except REVIEWER_FAILURES:
            return {}
        if message := review_message(verdict, sql, state["schema_context"]):
            return {"errors": [{"stage": "review", "message": message}]}
        return {}

    return review
