"""Synthesize: the chatbot's one answer writer, shared by both lanes.

Whatever the route, it gets the same input: a list of tasks and their results.
  sql_query route  1 task: the router's standalone question and the sql_query lane's result.
  analysis route   the planner's tasks and their sql_query lane results.
One LLM call (gpt-4o-mini). It sees the standalone question, every task's result (at most
MAX_ROWS_SHOWN rows each) and the last HISTORY_MESSAGES messages of the conversation.

check_grounding (plain code) finds every number in the answer and looks for it among the
allowed numbers (Option B, agreed with the project owner):
  - numbers in this turn's results (rows, including numbers inside text like '2017-01'),
  - numbers in the user's question, the task questions and the tasks' assumptions,
  - numbers in earlier answers in the history that were themselves grounded
    (AIMessage.additional_kwargs["grounded"] is True), and in earlier user questions.
Small whole numbers 0-10 are ignored (counting words like "3 sellers").
If numbers are unsupported, synthesize is asked ONCE more with the list. If they are still
unsupported, the answer is kept, marked grounded = False, and the numbers are listed in
`ungrounded_numbers`.
"""
import re

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from chatbot.state import HISTORY_MESSAGES, ChatbotState, render_history

MAX_ROWS_SHOWN = 30
SMALL_INTEGER = 10                 # whole numbers up to this are not checked


class SynthesizedAnswer(BaseModel):
    answer: str = Field(description="The answer for the user, in plain language.")


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You write the final answer to a user's question about Olist, a Brazilian \
e-commerce marketplace. The question was answered by one or more database queries (tasks); \
you get every task's result.

# Rules
1. Use ONLY numbers that appear in the task results or in earlier answers of the \
conversation. Never calculate new numbers yourself (no differences, ratios, percentages or \
growth rates that are not in the results). You may round a number (1,234,567.89 -> about \
1.23 million) and you may describe a comparison in words ("higher", "about double").
2. Answer every part of the question. If a task failed or the data cannot answer it, say \
plainly which part could not be answered and why (use its reason if given). If a result \
has no matching data, say that nothing matched; do not guess.
3. Mention assumptions that change the meaning (for example "revenue excludes freight", \
"'this year' means 2018, the latest year in the data").
4. For "why" questions, describe what the numbers show (which states, categories or months \
stand out); do not invent causes the data does not show.
5. Money is in Brazilian reais (R$). Be concise: a short paragraph or a few bullet points. \
Do not mention SQL, tasks or queries; talk about the data.
6. Use the earlier conversation only for continuity and comparisons with earlier answers."""


def _render_value(v) -> str:
    return "NULL" if v is None else str(v)


def render_results(tasks: list[dict], results: dict[str, dict]) -> str:
    """Every task's question, outcome, assumptions and (at most MAX_ROWS_SHOWN) rows."""
    blocks = []
    for t in tasks:
        r = results.get(t["id"])
        lines = [f"## Task {t['id']}: {t['question']}"]
        if r is None:
            lines.append("Outcome: not run.")
        elif r["status"] == "unanswerable":
            lines.append(f"Outcome: the data cannot answer this. Reason: {r['unanswerable_reason']}")
        elif r["status"] != "ok":
            lines.append("Outcome: could not be answered (the query failed).")
        else:
            rows = r["rows"]
            if r["assumptions"]:
                lines.append("Assumptions: " + "; ".join(r["assumptions"]))
            if not rows:
                lines.append("Result: no matching data (0 rows).")
            else:
                lines.append(" | ".join(r["columns"]))
                lines += [" | ".join(_render_value(v) for v in row) for row in rows[:MAX_ROWS_SHOWN]]
                more = len(rows) - MAX_ROWS_SHOWN
                if more > 0 or r.get("truncated"):
                    total = f"{len(rows):,}{'+' if r.get('truncated') else ''}"
                    lines.append(f"(showing {min(len(rows), MAX_ROWS_SHOWN)} of {total} rows)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_synth_user_message(question: str, tasks: list[dict], results: dict[str, dict],
                             history: list[AnyMessage], unsupported: list[str]) -> str:
    parts = []
    if history:
        parts.append(f"Conversation so far:\n{render_history(history)}")
    parts.append(f"Question: {question}")
    parts.append(f"Task results:\n{render_results(tasks, results)}")
    if unsupported:
        parts.append("Your previous answer used numbers that do not appear in the results or "
                     f"earlier answers: {', '.join(unsupported)}. Rewrite the answer using only "
                     "numbers that appear there (describe comparisons in words instead).")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM call

def synthesize(question: str, tasks: list[dict], results: dict[str, dict],
               history: list[AnyMessage], unsupported: list[str],
               llm: BaseChatModel) -> SynthesizedAnswer:
    """Ask the LLM for one answer combining the task results. One LLM call, no other logic.

    Input:
        question: the user's question for this turn, as the router made it standalone.
        tasks: [{"id": "A", "question": "..."}]: one task (sql_query route) or the
            analysis planner's standalone sub-questions.
        results: {task id: {"question", "status" (ok | failed | unanswerable), "columns",
            "rows", "truncated", "assumptions", "unanswerable_reason", "sql", "error"}}.
            At most MAX_ROWS_SHOWN rows per task are shown to the LLM.
        history: the earlier conversation, oldest first (HumanMessage / AIMessage); the
            node passes the last HISTORY_MESSAGES.
        unsupported: [] on the first call; on the retry, the numbers the previous answer
            used that are not in the results or earlier grounded answers.
        llm: a LangChain chat model, e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0).

    Output: SynthesizedAnswer(answer=...).

    What a good answer looks like:
        - It answers every part of the question, using only numbers from the task
          results or earlier answers (rounding allowed; no new differences, ratios or
          growth rates it computed itself).
        - It says plainly which parts could not be answered and why.
        - It mentions assumptions that change the meaning (revenue excludes freight,
          'this year' = 2018, ...).
        - "Why" questions get what the numbers show, not invented causes.
        - It is short, in plain language, with money in R$, and never mentions SQL.
        - On the retry, the flagged numbers are gone.

    Raises OutputParserException / ValidationError if the response can't be used; the
    node then retries once and finally falls back to a plain list of what was answered.
    """
    structured = llm.with_structured_output(SynthesizedAnswer, method="json_schema", strict=True)
    return structured.invoke([SystemMessage(SYSTEM_PROMPT),
                              HumanMessage(build_synth_user_message(question, tasks, results,
                                                                    history, unsupported))])


# ---------------------------------------------------------------- grounding check (code)

_NUMBER = re.compile(
    r"(?<![\w.])"                                   # not inside a word or after a dot
    r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"         # 1,234,567 / 1234 / with decimals
    r"(?:\s*((?i:million|thousand|billion|bn|mn)|[kKM]))?"
    r"(?!\w)")                                      # not the start of an id like 1025f0e2...
_SCALE = {"thousand": 1e3, "k": 1e3, "K": 1e3, "million": 1e6, "mn": 1e6, "M": 1e6,
          "billion": 1e9, "bn": 1e9}


def extract_numbers(text: str) -> list[tuple[str, float, float]]:
    """[(as written, value, tolerance)] for every number in `text`.

    Tolerance = half a unit of the last written digit, times the scale word, so "12.5
    million" covers 12,450,000-12,550,000 and "4.1" covers 4.05-4.15 (at least 0.01).
    """
    out = []
    for m in _NUMBER.finditer(text or ""):
        whole, decimals, scale_word = m.group(1), m.group(2) or "", m.group(3)
        value = float(whole.replace(",", "") + (f".{decimals}" if decimals else ""))
        scale = _SCALE.get(scale_word, 1.0) if scale_word else 1.0
        tol = max(0.01, 0.5 * 10 ** (-len(decimals)) * scale)
        out.append((m.group(0).strip(), value * scale, tol))
    return out


def _numbers_in_value(v) -> list[float]:
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    return [n for _, n, _ in extract_numbers(str(v))]


def allowed_numbers(question: str, tasks: list[dict], results: dict[str, dict],
                    history: list[AnyMessage]) -> list[float]:
    """Every number the answer may use (Option B)."""
    allowed = [n for _, n, _ in extract_numbers(question)]
    for t in tasks:
        allowed += [n for _, n, _ in extract_numbers(t["question"])]
    for r in results.values():
        for a in r.get("assumptions", []):
            allowed += [n for _, n, _ in extract_numbers(a)]
        for row in r.get("rows", []):
            for v in row:
                allowed += _numbers_in_value(v)
    for m in history:
        grounded_answer = isinstance(m, AIMessage) and m.additional_kwargs.get("grounded") is True
        if isinstance(m, HumanMessage) or grounded_answer:
            allowed += [n for _, n, _ in extract_numbers(str(m.content))]
    return allowed


def check_grounding(answer: str, allowed: list[float]) -> list[str]:
    """Numbers in `answer` (as written) that match no allowed number. [] = grounded.

    Signs are ignored: text extraction reads "-3.2" as 3.2, and "fell by 3.2" is fine.
    """
    unsupported = []
    allowed_abs = [abs(a) for a in allowed]
    for written, value, tol in extract_numbers(answer):
        if value == int(value) and abs(value) <= SMALL_INTEGER and "." not in written:
            continue                                # counting words: "3 sellers", "top 5"
        if not any(abs(value - a) <= tol + 1e-9 for a in allowed_abs):
            unsupported.append(written)
    return list(dict.fromkeys(unsupported))


# ---------------------------------------------------------------- node

def _fallback_answer(tasks: list[dict], results: dict[str, dict]) -> str:
    """Used only if the LLM response can't be used twice: says what was answered, no numbers."""
    lines = ["I couldn't write a summary for this question. Here is what was looked up:"]
    for t in tasks:
        status = (results.get(t["id"]) or {}).get("status", "not run")
        lines.append(f"- {t['question']} ({status})")
    return "\n".join(lines)


def final_status(results: dict[str, dict]) -> str:
    statuses = [r["status"] for r in results.values()]
    if any(s == "ok" for s in statuses):
        return "ok"
    if statuses and all(s == "unanswerable" for s in statuses):
        return "unanswerable"
    return "failed"


def make_synthesize(llm: BaseChatModel):
    """Build the synthesize node around `llm` (e.g. ChatOpenAI(model="gpt-4o-mini", temperature=0))."""

    def synthesize_node(state: ChatbotState) -> dict:
        question = state.get("standalone_question") or state["question"]
        tasks, results = state["tasks"], state.get("results", {})
        attempt = state.get("synth_attempts", 0) + 1
        history = state.get("messages", [])[:-1][-HISTORY_MESSAGES:]
        previous_unsupported = state.get("ungrounded_numbers", []) if attempt == 2 else []

        try:
            answer = synthesize(question, tasks, results, history, previous_unsupported, llm).answer
            unsupported = check_grounding(answer, allowed_numbers(question, tasks, results, history))
        except (OutputParserException, ValidationError):
            if attempt == 1:
                return {"synth_attempts": 1, "ungrounded_numbers": []}     # try once more
            answer, unsupported = _fallback_answer(tasks, results), []

        if unsupported and attempt == 1:
            # not final yet: routing sends it back to synthesize with the list
            return {"answer": answer, "synth_attempts": 1, "ungrounded_numbers": unsupported}

        return {"answer": answer, "synth_attempts": attempt, "ungrounded_numbers": unsupported,
                "status": final_status(results),
                "messages": [AIMessage(content=answer,
                                       additional_kwargs={"grounded": not unsupported})]}

    return synthesize_node
