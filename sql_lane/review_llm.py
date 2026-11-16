"""Review, layer 2: the LLM check, only for rules that depend on the question.

Runs only on SQL that passed review_code.code_review, so the SQL is safe, uses real tables
and columns, and already follows every rule code can check from the SQL alone (revenue
filter, latest review, delivery filters, rounding, percentages, ...). What is left needs
the question: does LIMIT 5 match "top 5", is "where customers live" answered with
customers.state, does the query answer what was asked at all.

llm_review()     one LLM call -> ReviewVerdict (the function for the DeepEval tests)
review_message() turns the verdict into feedback for Generate SQL, or None to approve.
                 Every quote is checked against the SQL: a rejection with nothing real
                 behind it is ignored, and an approval whose own check found nothing (or
                 invented its proof) is overruled.
"""
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from sql_lane.generate import render_data_facts, render_schema_context
from sql_lane.rules import AMBIGUITY_DEFAULTS

# The rules only the LLM can check, because they depend on what the question asks.
# Ids match the rule numbers in rules.py (C = global rules, SQL = SQL rules).
# Each: (applies when the question ..., what the SQL must do).
QUESTION_RULES: dict[str, tuple[str, str]] = {
    "C2": ("asks how many units / items / products were sold",
           "count order_items rows with COUNT(*); there is no quantity column."),
    "C3": ("asks for a number of orders",
           "count from orders, never from order_items."),
    "C4": ("asks for a number of customers",
           "use COUNT(DISTINCT customer_id)."),
    "C8": ("says where customers live, are from or are based ('live', 'from', 'home', "
           "'customer state', 'customer city')",
           "group or filter by customers.state / customers.city. orders.delivery_state and "
           "orders.delivery_city are where one order was shipped, NOT where a customer lives, "
           "so using them here breaks C8. (For where orders went, delivery_state is correct.)"),
    "C9": ("asks about an area, region or 'where' without naming a city",
           "group by a state column, not by a city column."),
    "C11": ("asks about conversion of leads",
            "use leads LEFT JOIN closed_deals on mql_id; rate = deals / leads."),
    "C13": ("asks to search review text for words",
            "use Portuguese words."),
    "SQL7": ("asks for 'top N', 'most', 'least', 'best', or 'by month' / over time",
             "ORDER BY that matches; for 'top N' also LIMIT N with N exactly as in the "
             "question ('top' alone = 10)."),
    "SQL6": ("names a specific value to filter on, such as a category, status, state, payment "
             "method, channel, lead type or business type (e.g. 'health and beauty', "
             "'cancelled orders', 'paid by card', 'São Paulo')",
             "every value compared with =, <> or IN on a column that has a 'Values:' list in the "
             "schema must be copied exactly from that list: 'health_beauty' not 'Health & "
             "Beauty', 'canceled' not 'cancelled', 'credit_card' not 'card', 'SP' not 'São "
             "Paulo'. A value not in the list silently matches nothing. ILIKE patterns and "
             "columns without a 'Values:' list (e.g. city) are exempt."),
}
QuestionRuleId = Literal["C2", "C3", "C4", "C8", "C9", "C11", "C13", "SQL6", "SQL7"]
RuleId = Literal["C2", "C3", "C4", "C8", "C9", "C11", "C13", "SQL6", "SQL7",
                 "does_not_answer_question"]


class RuleCheck(BaseModel):
    """One rule the question triggers, and what the SQL does about it."""
    rule: QuestionRuleId
    question_words: str = Field(
        description="The words copied exactly from the question that make this rule apply.")
    sql_text: str | None = Field(
        description="The exact SQL text that meets the rule, or null if nothing in the query does.")
    listed_value: str | None = Field(
        description="SQL6 only: the value copied exactly from that column's 'Values:' list in the "
                    "schema that the question means (e.g. canceled for 'cancelled orders'). Null "
                    "for every other rule.")


class ReviewVerdict(BaseModel):
    """What the reviewer returns, in this order: it checks before it decides."""
    checks: list[RuleCheck] = Field(
        description="Step 1: one entry per rule the question triggers; empty if none.")
    answers_question: str = Field(
        description="Step 2: what the query returns (metric, grouping, filters, period) "
                    "compared with what the question asks.")
    approved: bool = Field(description="Step 3: false only if a check has sql_text null or "
                                       "wrong, or the query does not answer the question.")
    rule_broken: RuleId | None = Field(description="The rule broken; null when approved.")
    evidence: str | None = Field(
        description="When rejecting: text copied exactly from the SQL that is wrong, or the "
                    "clause where the missing part belongs. Null when approved.")
    explanation: str = Field(
        description="When rejecting: what to change, in one or two sentences. Empty when approved.")

    @model_validator(mode="after")
    def _consistent(self):
        if not self.approved and (self.rule_broken is None or not self.evidence):
            raise ValueError("approved is false, so rule_broken and evidence are required")
        if self.approved:
            self.rule_broken, self.evidence = None, None
        return self


def render_question_rules() -> str:
    return "\n".join(f"{rid}. Applies when the question {when}. Then: {then}"
                      for rid, (when, then) in QUESTION_RULES.items())


def build_review_system_prompt(schema_context: dict[str, dict]) -> str:
    return f"""You check whether one PostgreSQL query answers the question it was written for. \
The data is Olist, a Brazilian e-commerce marketplace.

Code has already checked everything that does not depend on the question: the query is \
safe, uses real tables and columns, and follows the data rules (revenue filters, latest \
review only, delivery filters, no row multiplication, rounding, percentages 0-100). Do not \
check those again. Check only the rules below and whether the query answers the question.

# Procedure (follow it exactly)
1. checks: go through the rules below. A rule applies only when the question itself \
contains what its "Applies when" describes; copy those exact words from the question into \
question_words. If the question has no such words, the rule does not apply: leave it out. \
For each rule that applies, copy the exact SQL text that meets it into sql_text, searching \
the whole query (every CTE and subquery), or null if nothing does.
   SQL6 is checked for EVERY query, whatever the question says: for each quoted value the \
SQL compares with =, <> or IN, add a check with rule SQL6, question_words = the words in the \
question that value stands for, sql_text = that comparison, and listed_value = the value \
copied exactly from that column's "Values:" list in the schema that the question means \
(for 'cancelled orders' it is canceled; for 'health and beauty' it is health_beauty). If the \
quoted value in the SQL is not exactly listed_value, SQL6 is broken, even if it means the \
same thing.
2. answers_question: say what the query returns (metric, grouping, filters, period) and \
whether that is what the question asks, including every part of the question.
3. approved: false only if a check has sql_text null or plainly wrong (including an SQL6 \
value missing from its Values list), or the query returns \
a different metric, grouping, filter or period than asked, or leaves out part of the \
question (rule does_not_answer_question). Otherwise true.
4. When rejecting: rule_broken = the rule; evidence = text copied character for character \
from the SQL that is wrong, or the clause where the missing part belongs (e.g. the ORDER \
BY line for a missing LIMIT); explanation = what to change.

# Never reject for
- A choice from the ambiguity defaults, or a reasonable assumption the writer stated.
- Anything code already checked (listed above), or a requirement not in the question.
- Style: formatting, alias wording, CTE vs subquery, extra columns, a trailing semicolon.
- An order counting once in each of its categories, or reaching products from reviews or \
payments through order_items.
Every false rejection wastes one of only three attempts. When unsure, approve.

# Rules
{render_question_rules()}

# Data facts
{render_data_facts()}

# Ambiguity defaults
{AMBIGUITY_DEFAULTS}

# Schema (the tables the query may use)
{render_schema_context(schema_context)}"""


def build_review_user_message(question: str, sql: str, assumptions: list[str]) -> str:
    listed = "\n".join(f"- {a}" for a in assumptions) or "- (none)"
    return f"Question: {question}\n\nAssumptions made by the writer:\n{listed}\n\nSQL:\n{sql}"


def llm_review(question: str, sql: str, assumptions: list[str],
               schema_context: dict[str, dict], llm: BaseChatModel) -> ReviewVerdict:
    """Ask the LLM whether `sql` answers `question`. One LLM call, no other logic.

    Only called for SQL that already passed code_review: safe, real tables and columns,
    and following every rule checkable from the SQL alone (revenue filter, latest
    review, delivery filters, row multiplication, rounding, percentages).

    Input:
        question: one complete plain-English question about the Olist data.
        sql: the PostgreSQL query Generate SQL wrote for it.
        assumptions: Generate SQL's assumptions, e.g. ["'Top' means top 10"].
        schema_context: from schema select: the selected tables with columns,
            descriptions, allowed values and joins.
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0).

    Output: ReviewVerdict with
        checks: one {rule, question_words, sql_text, listed_value} per rule the question
            triggers (listed_value: SQL6 only, the exact listed value meant), from
            C2 (units), C3 (order counts), C4 (customer counts), C8 (where customers
            live = customers.state), C9 (state unless a city is named), C11
            (conversion), C13 (Portuguese keywords), SQL6 (filter values copied exactly
            from the schema's Values lists), SQL7 (ORDER BY / LIMIT N matching the
            question). question_words = the words from the question that trigger
            it; sql_text = the SQL text that meets it, or null.
        answers_question: what the query returns compared with what the question asks.
        approved: True if the query answers the question.
        rule_broken: when rejected, one of the rules above or does_not_answer_question;
            null when approved.
        evidence: when rejected, text copied exactly from the SQL that is wrong or where
            the missing part belongs; null when approved. (See review_message: a check
            whose sql_text is null or not really in the SQL counts as failed.)
        explanation: when rejected, what to change; empty when approved.

    What a good verdict looks like:
        - Rejects "top 5" with no LIMIT or LIMIT 10 (SQL7), "by month" with no ORDER BY
          (SQL7), "where customers live" answered with orders.delivery_state (C8), an
          area question grouped by city when no city was named (C9), a filter value not
          in the schema's Values list such as status = 'cancelled' or category =
          'Health & Beauty' (SQL6), a different metric,
          period, grouping or filter than asked, or a part of the question left out
          (does_not_answer_question).
        - Approves correct SQL, including documented defaults ("top" = 10, "area" =
          state, "this year" = latest year in the data) and reasonable stated
          assumptions. Never rejects over style or rules code already checked.
        - Only rules the question triggers are checked (no customer-count rule for a
          review-score question); sql_text and evidence are text really in the SQL;
          the explanation says how to fix it.

    Raises OutputParserException / ValidationError (unusable response),
    OpenAIRefusalError or LengthFinishReasonError; the node then approves.
    """
    structured = llm.with_structured_output(ReviewVerdict, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(build_review_system_prompt(schema_context)),
                              HumanMessage(build_review_user_message(question, sql, assumptions))])


def _quote_in_sql(quote: str | None, sql: str) -> bool:
    """True if `quote` appears in `sql`, ignoring case and whitespace differences."""
    def norm(x: str) -> str:
        return " ".join(x.lower().split())
    return bool(quote) and norm(quote) in norm(sql)


def _listed_values(schema_context: dict[str, dict]) -> set[str]:
    """Every value in a 'Values:' list of the selected tables."""
    return {v for t in schema_context.values() for c in t["columns"] for v in c.get("values", [])}


def _wrong_value(check: RuleCheck, listed: set[str], sql: str) -> bool:
    """SQL6: the reviewer named the listed value the question means (on whichever check),
    and that exact quoted value appears nowhere in the SQL. Only trusted when listed_value
    really is in a Values list of the selected tables."""
    return (bool(check.listed_value) and check.listed_value in listed
            and f"'{check.listed_value}'" not in sql)


def review_message(verdict: ReviewVerdict, sql: str,
                   schema_context: dict[str, dict] | None = None) -> str | None:
    """Feedback for Generate SQL, or None to approve. Reads the verdict; never calls the LLM.

    A rule check fails when its sql_text is null (nothing meets the rule), is not really in
    the SQL (the reviewer invented the proof), or, for SQL6, the listed_value the reviewer
    picked from the schema's Values list appears nowhere in the SQL (code compares exactly,
    which the LLM does badly: 'cancelled' vs 'canceled'). Then:
    - Rejected: kept if the evidence is really in the SQL, or a check of the broken rule
      failed. An SQL6 rejection is kept only if code finds the value mismatch. A rejection
      with nothing real behind it is ignored.
    - Approved: overruled if a check failed, since the reviewer's own check says the rule
      is not met.
    """
    listed = _listed_values(schema_context or {})
    failed = [c for c in verdict.checks
              if not _quote_in_sql(c.sql_text, sql) or _wrong_value(c, listed, sql)]

    if not verdict.approved:
        if verdict.rule_broken == "SQL6":     # code can settle SQL6 exactly: trust only a mismatch
            backed = any(_wrong_value(c, listed, sql) for c in verdict.checks)
        else:
            backed = _quote_in_sql(verdict.evidence, sql) or \
                any(c.rule == verdict.rule_broken for c in failed)
        if backed:
            label = ("does not answer the question"
                     if verdict.rule_broken == "does_not_answer_question"
                     else f"Rule {verdict.rule_broken} broken")
            return f"LLM review: {label}: {verdict.explanation}"
        if not failed:
            return None                     # unsupported rejection

    if failed:
        c = failed[0]
        if _wrong_value(c, listed, sql):
            return (f"LLM review: Rule SQL6 broken: the SQL never uses '{c.listed_value}', the "
                    f"value in the schema's Values list for '{c.question_words}'; filter with "
                    f"exactly that value.")
        return f"LLM review: Rule {c.rule} broken: the question says '{c.question_words}', " \
               f"so {QUESTION_RULES[c.rule][1]}"
    return None
