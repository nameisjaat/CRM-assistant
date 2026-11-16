"""Evaluate one attempt of the sql_query lane, from Generate SQL to the result.

    .venv\\Scripts\\python.exe eval\\eval_generate.py

For every question in eval/sql_lane_questions.yaml, one attempt runs in the lane's order:

    generate -> review_code -> review_llm -> execute -> validate -> compare with golden

  - schema_context is built from the golden `tables`, so schema select's own mistakes
    (measured by eval_schema_select.py) don't count against Generate.
  - SQL that fails review_code is NEVER executed (same safety rule as the lane).
  - SQL the LLM reviewer rejects IS still executed (it already passed every code check):
    the pipeline still counts it as failed at review_llm, but its result shows whether
    the rejection was right (true catch) or wrong (false rejection).
  - One attempt = the last attempt, so an empty result is accepted (as the lane does on
    its last attempt).

Metrics (DeepEval):
  code   Answerability       answerable matches the golden                  (all questions)
  code   Pipeline Accuracy   passed every stage AND the result matches      (answerable)
  code   SQL Result Match    the generated SQL's result matches, whatever   (answerable)
                             the reviewer said
  code   Shape Match         expected_shape kind, max_rows, order_matters   (answerable)
  GEval  Assumptions Quality the stated assumptions cover the golden ones   (answerable)
  GEval  Refusal Quality     unanswerable_reason names the real limitation  (unanswerable)

Cost per run: ~30 generate calls (gpt-4o), up to ~26 reviewer calls (gpt-4o-mini),
~30 judge calls, up to 26 read-only database queries. Needs OPENAI_API_KEY and
DATABASE_URL in the project's .env.
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # so `sql_lane` imports when run as a script

import psycopg2
import psycopg2.errors
import yaml
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.metrics import BaseMetric, GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from eval.result_match import results_match
from sql_lane.catalog import build_schema_context
from sql_lane.execute import TIMEOUT_MESSAGE, SharedConnection
from sql_lane.generate import generate_sql
from sql_lane.review import REVIEWER_FAILURES
from sql_lane.review_code import code_review
from sql_lane.review_llm import llm_review, review_message
from sql_lane.validate import check_result, is_empty

load_dotenv(ROOT / ".env")

GOLDEN_PATH = ROOT / "eval" / "sql_lane_questions.yaml"
GENERATE_MODEL = "gpt-4o"                # the lane's generate model (sql_lane/graph.py)
REVIEW_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 1.0                          # code metrics: pass only when fully right
GEVAL_THRESHOLD = 0.7
ONLY_VERIFIED = True

STAGES = ["generate", "review_code", "review_llm", "execute", "validate"]


# ---------------------------------------------------------------- 1. load goldens

def load_goldens(path: Path = GOLDEN_PATH) -> list[dict]:
    goldens = yaml.safe_load(path.read_text(encoding="utf-8"))
    if ONLY_VERIFIED:
        goldens = [g for g in goldens if g.get("verified")]
    return goldens


# ---------------------------------------------------------------- 2. one attempt per question

def run_one(g: dict, generate_llm, review_llm, conn: SharedConnection) -> dict:
    """One attempt through the lane for golden question `g`. Returns a flat record."""
    ctx = build_schema_context(g["tables"])
    r = {"id": g["id"], "question": g["question"], "difficulty": g["difficulty"],
         "rules": g.get("rules", []), "expected_answerable": g["answerable"],
         "answerable": None, "sql": None, "shape": None, "assumptions": [],
         "unanswerable_reason": None, "failed_stage": None, "error": None,
         "reviewer": None, "executed": False, "exec_error": None,
         "result_match": False, "result_reason": "not executed"}

    # generate
    try:
        out = generate_sql(g["question"], ctx, [], generate_llm)
    except (OutputParserException, ValidationError) as e:
        r.update(failed_stage="generate", error=str(e).strip()[:300])
        return r
    r.update(answerable=out.answerable, assumptions=out.assumptions,
             unanswerable_reason=out.unanswerable_reason)
    if not out.answerable:
        return r
    sql = out.sql.strip()
    shape = out.expected_shape.model_dump()
    r.update(sql=sql, shape=shape)

    # review_code: failing SQL is never executed
    if error := code_review(sql):
        r.update(failed_stage="review_code", error=error)
        return r

    # review_llm (an unusable reviewer answer approves, as in the lane)
    try:
        verdict = llm_review(g["question"], sql, out.assumptions, ctx, review_llm)
        message = review_message(verdict, sql, ctx)
    except REVIEWER_FAILURES:
        message = None
    r["reviewer"] = "rejected" if message else "approved"
    if message:
        r.update(failed_stage="review_llm", error=message)

    # execute: also after an LLM rejection (safe: code checks passed), to judge the reviewer
    try:
        columns, rows, truncated = conn.run(sql)
    except psycopg2.errors.QueryCanceled:
        r["exec_error"] = TIMEOUT_MESSAGE
    except psycopg2.Error as e:
        r["exec_error"] = (getattr(e, "pgerror", None) or str(e)).strip()
    if r["exec_error"]:
        r["result_reason"] = f"execute error: {r['exec_error']}"
        if r["failed_stage"] is None:
            r.update(failed_stage="execute", error=r["exec_error"])
        return r
    r["executed"] = True

    # validate (one attempt is the last attempt: an empty result is accepted)
    if r["failed_stage"] is None and not is_empty(rows):
        if error := check_result(columns, rows, truncated, shape, sql):
            r.update(failed_stage="validate", error=error)

    # compare with the golden result
    if g["answerable"]:
        match, reason = results_match(g["expected_result"], columns, rows,
                                      g["expected_shape"]["order_matters"])
        r.update(result_match=match, result_reason=reason)
    return r


def score(r: dict, g: dict) -> dict:
    """Scores and reasons for the code metrics, stored in the test case metadata."""
    s = {}
    ok = r["answerable"] == g["answerable"]
    s["answerability"] = (1.0 if ok else 0.0,
                          f"expected answerable={g['answerable']}, got {r['answerable']}"
                          + (f" (failed at generate: {r['error']})" if r["failed_stage"] == "generate" else ""))
    if g["answerable"]:
        correct = r["failed_stage"] is None and r["answerable"] is True and r["result_match"]
        s["pipeline"] = (1.0 if correct else 0.0,
                         "passed every stage and the result matches" if correct else
                         f"failed at {r['failed_stage']}: {r['error']}" if r["failed_stage"] else
                         f"passed every stage but the result is wrong: {r['result_reason']}")
        s["sql_result"] = (1.0 if r["result_match"] else 0.0, r["result_reason"])
        exp, got = g["expected_shape"], r["shape"] or {}
        parts = [k for k in ("kind", "max_rows", "order_matters") if got.get(k) == exp[k]]
        s["shape"] = (len(parts) / 3,
                      f"expected {exp}, got {r['shape']}" if len(parts) < 3 else "shape matches")
    return s


# ---------------------------------------------------------------- 3. metrics

class RecordMetric(BaseMetric):
    """A code metric whose score was computed in score(); reads it from test_case.metadata."""

    def __init__(self, name: str, key: str, threshold: float = THRESHOLD):
        self.name, self.key = name, key
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.evaluation_model = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score, self.reason = test_case.metadata["scores"][self.key]
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return self.name


def geval_metrics() -> tuple[GEval, GEval]:
    assumptions = GEval(
        name="Assumptions Quality",
        evaluation_steps=[
            "The expected output lists the defaults or interpretations a correct answer to the "
            "question must use. The actual output lists the assumptions the SQL writer stated.",
            "For each expected item, check that the actual output states the same assumption "
            "(same meaning, any wording).",
            "Penalize each expected item that is missing from the actual output or stated with a "
            "different meaning.",
            "Penalize any actual assumption that contradicts the question or an expected item "
            "(for example revenue including freight, or 'this year' meaning a year other than 2018).",
            "Extra assumptions that are reasonable and consistent with the question must NOT lower "
            "the score.",
            "If the expected output says no specific defaults are required, score high unless an "
            "actual assumption is wrong.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Every expected assumption is stated; nothing contradicts the question."),
            Rubric(score_range=(5, 8), expected_outcome="Most expected assumptions are stated, or one is vague; nothing clearly wrong."),
            Rubric(score_range=(0, 4), expected_outcome="Key expected assumptions are missing, or a stated assumption is wrong."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        threshold=GEVAL_THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )
    refusal = GEval(
        name="Refusal Quality",
        evaluation_steps=[
            "The question cannot be answered from the data. The actual output should be a refusal "
            "explaining why.",
            "Check that the refusal names the limitation described in the expected output (same "
            "meaning, any wording).",
            "Penalize a refusal that gives a wrong or vague reason, invents data, or claims the data "
            "contains something it does not.",
            "If the actual output shows an answer or SQL was produced instead of a refusal, the "
            "score must be 0-2.",
            "Do not judge wording style or length.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="A clear refusal naming the right limitation."),
            Rubric(score_range=(5, 8), expected_outcome="A refusal with a vague or partly right reason."),
            Rubric(score_range=(0, 4), expected_outcome="No refusal, or a wrong reason."),
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        threshold=GEVAL_THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )
    return assumptions, refusal


# ---------------------------------------------------------------- 4. run and evaluate

def run(generate_llm, review_llm, conn: SharedConnection) -> list[dict]:
    goldens = load_goldens()
    records, answerable_cases, unanswerable_cases = [], [], []

    for i, g in enumerate(goldens, 1):
        print(f"[{i}/{len(goldens)}] {g['id']} {g['question']}")
        r = run_one(g, generate_llm, review_llm, conn)
        r["scores"] = score(r, g)
        records.append(r)
        meta = {"id": g["id"], "scores": r["scores"]}

        if g["answerable"]:
            stated = "\n".join(f"- {a}" for a in r["assumptions"]) or "(no assumptions stated)"
            expected = "\n".join(f"- {a}" for a in g.get("expected_assumptions", [])) \
                or "(no specific defaults are required)"
            answerable_cases.append(LLMTestCase(input=g["question"], actual_output=stated,
                                                expected_output=expected, metadata=meta))
        else:
            if r["answerable"] is False:
                given = r["unanswerable_reason"] or "(refused with no reason)"
            elif r["answerable"] is True:
                given = f"(no refusal: the model answered with SQL)\n{r['sql']}"
            else:
                given = f"(no usable response: {r['error']})"
            unanswerable_cases.append(LLMTestCase(
                input=g["question"], actual_output=given,
                expected_output=f"The data cannot answer this. The refusal should mention: "
                                f"{g['unanswerable_reason_topic']}.",
                metadata=meta))

    # No DeepEval result cache: every run scores fresh, and the cache files (.deepeval/) can't
    # be locked by OneDrive sync or by overlapping async writes (that caused
    # "'NoneType' object has no attribute 'test_cases_lookup_map'").
    no_cache = CacheConfig(write_cache=False, use_cache=False)
    assumptions_metric, refusal_metric = geval_metrics()
    if answerable_cases:
        evaluate(test_cases=answerable_cases, metrics=[
            RecordMetric("Answerability", "answerability"),
            RecordMetric("Pipeline Accuracy", "pipeline"),
            RecordMetric("SQL Result Match", "sql_result"),
            RecordMetric("Shape Match", "shape"),
            assumptions_metric,
        ], cache_config=no_cache)
    if unanswerable_cases:
        evaluate(test_cases=unanswerable_cases, metrics=[
            RecordMetric("Answerability", "answerability"),
            refusal_metric,
        ], cache_config=no_cache)
    return records


# ---------------------------------------------------------------- 5. summary

def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


def print_summary(records: list[dict]) -> None:
    ans = [r for r in records if r["expected_answerable"]]
    una = [r for r in records if not r["expected_answerable"]]
    line = "=" * 80
    print(f"\n{line}\nONE ATTEMPT: GENERATE -> REVIEW -> EXECUTE -> VALIDATE "
          f"({len(records)} questions, {GENERATE_MODEL})\n{line}")

    print(f"Answerability:         {_pct(sum(r['scores']['answerability'][0] == 1 for r in records), len(records))}")
    pipeline_ok = sum(r["scores"]["pipeline"][0] == 1 for r in ans)
    sql_ok = sum(r["result_match"] for r in ans)
    print(f"Pipeline accuracy:     {_pct(pipeline_ok, len(ans))}   <- first-attempt accuracy")
    print(f"SQL result match:      {_pct(sql_ok, len(ans))}   <- generated SQL correct, whatever the reviewer said")
    print(f"Shape match:           {_pct(sum(r['scores']['shape'][0] == 1 for r in ans), len(ans))}")
    print(f"Unanswerable refused:  {_pct(sum(r['answerable'] is False for r in una), len(una))}")
    print("(Assumptions Quality and Refusal Quality: see the DeepEval results above.)")

    # funnel over the answerable questions, in the lane's order
    print("\nFunnel (answerable questions):")
    alive = [r for r in ans if r["answerable"] is True and r["sql"]]
    print(f"  generated SQL          {len(alive)}/{len(ans)}")
    for stage in STAGES[1:]:
        alive = [r for r in alive if r["failed_stage"] != stage]
        print(f"  passed {stage:15} {len(alive)}")
    print(f"  correct result         {sum(r['result_match'] for r in alive)}")

    # the LLM reviewer judged against the real results
    judged = [r for r in ans if r["reviewer"] and r["executed"]]
    tc = sum(r["reviewer"] == "rejected" and not r["result_match"] for r in judged)
    fr = sum(r["reviewer"] == "rejected" and r["result_match"] for r in judged)
    ms = sum(r["reviewer"] == "approved" and not r["result_match"] for r in judged)
    ca = sum(r["reviewer"] == "approved" and r["result_match"] for r in judged)
    print(f"\nLLM reviewer ({len(judged)} SQL that passed review_code and ran):")
    print(f"  true catches {tc}   false rejections {fr}   misses {ms}   correct approvals {ca}")

    print("\nPipeline accuracy by difficulty:")
    for level in ("easy", "medium", "hard"):
        rs = [r for r in ans if r["difficulty"] == level]
        if rs:
            print(f"  {level:7} {_pct(sum(r['scores']['pipeline'][0] == 1 for r in rs), len(rs))}")

    print("\nPipeline accuracy by rule tested:")
    by_rule = defaultdict(list)
    for r in ans:
        for rule in r["rules"]:
            by_rule[rule].append(r)
    for rule in sorted(by_rule):
        rs = by_rule[rule]
        print(f"  {rule:24} {_pct(sum(r['scores']['pipeline'][0] == 1 for r in rs), len(rs))}")

    misses = [r for r in ans if r["scores"]["pipeline"][0] < 1] + \
             [r for r in una if r["answerable"] is not False]
    print(f"\nMisses ({len(misses)}):" if misses else "\nMisses: none")
    for r in misses:
        print(f"\n  {r['id']} [{r['difficulty']}] {r['question']}")
        if not r["expected_answerable"]:
            print(f"     expected a refusal; got answerable={r['answerable']}")
        else:
            print(f"     {r['scores']['pipeline'][1]}")
            if r["failed_stage"] in ("review_llm",) and r["executed"]:
                print(f"     result if it had run: {'correct' if r['result_match'] else r['result_reason']}")
        if r["sql"]:
            print("     SQL: " + r["sql"].replace("\n", "\n          "))


def run_local() -> list[dict]:
    """Standalone convenience: build the models and the connection, then run."""
    generate_llm = ChatOpenAI(model=GENERATE_MODEL, temperature=0)
    review_llm = ChatOpenAI(model=REVIEW_MODEL, temperature=0)
    return run(generate_llm, review_llm, SharedConnection())


if __name__ == "__main__":
    print_summary(run_local())
