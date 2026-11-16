"""Router: picks exactly one route for the user's question and makes it standalone.

One LLM call (gpt-4o-mini). It sees the question and the last HISTORY_MESSAGES messages,
so a follow-up ("what about 2017?") becomes a complete question before any lane sees it.
  sql_query     one SQL query can answer it, however many tables it joins
  analysis      it needs several separate numbers, or reasoning over them
  out_of_scope  it is not about the Olist data (the chatbot replies without an LLM)
If the response can't be used, the question goes to sql_query as typed.
"""
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from chatbot.state import HISTORY_MESSAGES, ChatbotState, render_history


class Route(BaseModel):
    """What the LLM returns. `reasoning` comes first so it decides after thinking."""
    reasoning: str = Field(description="One or two sentences: what the user wants and why this route.")
    route: Literal["sql_query", "analysis", "out_of_scope"]
    standalone_question: str = Field(
        description="The question rewritten to be complete on its own, with every name, period "
                    "and filter from the conversation spelled out. Unchanged if already complete.")


SYSTEM_PROMPT = """You route questions for a chatbot that answers questions about Olist, a \
Brazilian e-commerce marketplace, from its database: orders, revenue, products and \
categories, sellers, customers, reviews, payments, deliveries, and the marketing funnel \
(leads and closed deals).

# Routes
- sql_query: ONE SQL query can return a result that directly answers it, however many \
tables it joins. Examples: "Top 10 sellers by revenue in 2018", "Revenue by customer state \
and category", "Conversion rate by marketing channel", "How many repeat customers are \
there?", "Average review score for the 5 categories with the most late deliveries" (one \
query can do that with a subquery).
- analysis: it needs several separate numbers, or reasoning over them. Examples: "How did \
revenue and orders change from 2017 to 2018, and why?", "Top sellers, and also the most \
common payment method" (two separate answers), "How is the marketing funnel performing?" \
(an overview), "Why did review scores drop?" (an explanation).
- out_of_scope: not a question about this data: greetings, small talk, general knowledge, \
requests to write or change data, or anything unrelated.

# Rules
1. Choose exactly one route. When unsure between sql_query and analysis, choose sql_query.
2. Questions the data cannot answer (product names, profit, forecasts) still go to \
sql_query or analysis: the lanes explain what is missing. Use out_of_scope only for \
questions that are not about the data at all.
3. standalone_question: rewrite follow-ups ("what about 2017?", "and for them?", "show \
the same by state") into a complete question using the conversation. Keep the user's \
wording for vague terms ("this year", "top", "area"). If the question is already complete, \
copy it unchanged."""


def build_router_user_message(question: str, history: list[AnyMessage]) -> str:
    parts = []
    if history:
        parts.append(f"Conversation so far:\n{render_history(history)}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def route_question(question: str, history: list[AnyMessage], llm: BaseChatModel) -> Route:
    """Ask the LLM which route answers `question`, and make it standalone. One LLM call.

    Input:
        question: the user's message for this turn, as typed (may be a follow-up).
        history: the earlier conversation, oldest first: HumanMessage (user) and AIMessage
            (assistant); the node passes the last HISTORY_MESSAGES.
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0).

    Output: Route with
        reasoning: a short explanation.
        route: "sql_query" | "analysis" | "out_of_scope".
        standalone_question: the question, complete on its own.

    What a good route looks like:
        - sql_query when one query answers it, however many tables it joins, including
          "for the top N X, show Y" questions one query can do with a subquery.
        - analysis for comparisons over several measures, overviews, "why" questions and
          questions with several separate parts.
        - out_of_scope only for messages not about the data (greetings, unrelated topics,
          requests to change data); questions about missing data (product names, profit,
          forecasts) still go to a lane.
        - standalone_question resolves follow-ups from the history ("what about 2017?"
          after "revenue by state in 2018" -> "What was revenue by state in 2017?") and
          keeps vague terms as the user wrote them; an already complete question is copied.

    Raises OutputParserException / ValidationError if the response can't be used; the
    node then sends the question to sql_query as typed.
    """
    structured = llm.with_structured_output(Route, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(SYSTEM_PROMPT),
                              HumanMessage(build_router_user_message(question, history))])


def make_router(llm: BaseChatModel):
    """Build the router node around `llm` (e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0))."""

    def router(state: ChatbotState) -> dict:
        question = state["question"]
        history = state.get("messages", [])[:-1][-HISTORY_MESSAGES:]   # before this question
        try:
            r = route_question(question, history, llm)
            standalone = r.standalone_question.strip() or question
            return {"route": r.route, "standalone_question": standalone}
        except (OutputParserException, ValidationError):
            return {"route": "sql_query", "standalone_question": question}

    return router
