"""Evaluate the whole sql_query lane, exactly as a user would run it.

    .venv\\Scripts\\python.exe eval\\eval_sql_pipeline.py

For every question in eval/sql_lane_questions.yaml: lane.invoke({"question": ...}) on the
compiled graph from build_sql_lane(): real schema select, up to 3 attempts (gpt-4o for
generate on every attempt), code + LLM review, execute, validate, the empty-result retry.
Everything is scored from the final state (status, tables, sql, errors, columns, rows,
assumptions, unanswerable_reason).

Metrics (DeepEval):
  code   Final Accuracy       answerable: status ok AND rows match the golden;
                              unanswerable: status unanswerable                  (all)
  code   Schema Recall        state.tables contains every golden table          (all)
  GEval  Assumptions Quality  as in eval_generate.py                             (answerable)
  GEval  Refusal Quality      as in eval_generate.py                             (unanswerable)
Summary adds: first-attempt accuracy and retry recovery (read from the same run's
`errors`), failure breakdown, attempts used, latency, by difficulty / by rule, misses.

Tokens and cost are not tracked here: LangSmith tracing records them per run.

Cost per run: per question 1 schema select call, 1-3 generate calls (gpt-4o),
up to 3 reviewer calls, 1 judge call, up to 3 read-only database queries. Needs
OPENAI_API_KEY and DATABASE_URL in the project's .env.
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # so `sql_lane` imports when run as a script

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.test_case import LLMTestCase

from eval_generate import RecordMetric, geval_metrics, load_goldens
from result_match import results_match
from sql_lane.execute import ConnectionFailure
from sql_lane.graph import build_sql_lane

load_dotenv(ROOT / ".env")

ASSUMPTIONS_THRESHOLD = 0.0              # Assumptions Quality: score shown, never a failure
REFUSAL_THRESHOLD = 0.0                  # Refusal Quality: score shown, never a failure


# ---------------------------------------------------------------- 1. run the lane

def run_one(lane, g: dict) -> dict:
    """One full lane run for golden question `g`. Returns a flat record."""
    start = time.time()
    try:
        state = lane.invoke({"question": g["question"]})       # fresh state per question
        crash = None
    except ConnectionFailure:
        raise                                                    # database down: stop the eval
    except Exception as e:                                       # keep going; record it
        state, crash = {}, f"{type(e).__name__}: {e}"
    seconds = time.time() - start

    errors = state.get("errors", [])
    r = {"id": g["id"], "question": g["question"], "difficulty": g["difficulty"],
         "rules": g.get("rules", []), "expected_answerable": g["answerable"],
         "golden_tables": list(g["tables"]), "seconds": seconds, "crash": crash,
         "status": state.get("status", "error" if crash else None),
         "tables": state.get("tables", []), "sqls": state.get("sql", []), "errors": errors,
         "assumptions": state.get("assumptions", []),
         "unanswerable_reason": state.get("unanswerable_reason"),
         "result_match": False, "result_reason": ""}
    r["attempts"] = len(r["sqls"])
    r["path"] = " -> ".join([e["stage"] for e in errors] + [str(r["status"])])

    if g["answerable"]:
        if r["status"] == "ok":
            match, reason = results_match(g["expected_result"], state.get("columns", []),
                                          state.get("rows", []), g["expected_shape"]["order_matters"])
            r.update(result_match=match, result_reason=reason)
        else:
            last = errors[-1] if errors else None
            r["result_reason"] = (crash or f"status {r['status']}"
                                  + (f", last error at {last['stage']}: {last['message']}" if last else ""))
    return r


def score(r: dict) -> dict:
    """Scores and reasons for the code metrics, stored in the test case metadata."""
    if r["expected_answerable"]:
        ok = r["status"] == "ok" and r["result_match"]
        final = (1.0 if ok else 0.0, f"status {r['status']}; {r['result_reason']}")
    else:
        ok = r["status"] == "unanswerable"
        final = (1.0 if ok else 0.0, f"expected status unanswerable, got {r['status']}")
    needed, chosen = set(r["golden_tables"]), set(r["tables"])
    recall = len(needed & chosen) / len(needed) if needed else 1.0
    missing = sorted(needed - chosen)
    return {"final": final,
            "schema_recall": (recall, f"missing: {', '.join(missing) or 'none'}")}


# ---------------------------------------------------------------- 2. run and evaluate

def run(lane) -> list[dict]:
    goldens = load_goldens()
    records, answerable_cases, unanswerable_cases = [], [], []

    for i, g in enumerate(goldens, 1):
        print(f"[{i}/{len(goldens)}] {g['id']} {g['question']}")
        r = run_one(lane, g)
        r["scores"] = score(r)
        records.append(r)
        print(f"         {r['path']}   ({r['seconds']:.1f}s)")
        meta = {"id": g["id"], "scores": r["scores"]}

        if g["answerable"]:
            stated = "\n".join(f"- {a}" for a in r["assumptions"]) or "(no assumptions stated)"
            expected = "\n".join(f"- {a}" for a in g.get("expected_assumptions", [])) \
                or "(no specific defaults are required)"
            answerable_cases.append(LLMTestCase(input=g["question"], actual_output=stated,
                                                expected_output=expected, metadata=meta))
        else:
            if r["status"] == "unanswerable":
                given = r["unanswerable_reason"] or "(refused with no reason)"
            elif r["sqls"]:
                given = f"(no refusal: the lane ended with status {r['status']} after SQL)\n{r['sqls'][-1]}"
            else:
                given = f"(no refusal: status {r['status']}{'; ' + r['crash'] if r['crash'] else ''})"
            unanswerable_cases.append(LLMTestCase(
                input=g["question"], actual_output=given,
                expected_output=f"The data cannot answer this. The refusal should mention: "
                                f"{g['unanswerable_reason_topic']}.",
                metadata=meta))

    # no DeepEval result cache (see eval_generate.py): every run scores fresh
    no_cache = CacheConfig(write_cache=False, use_cache=False)
    assumptions_metric, refusal_metric = geval_metrics()
    # pipeline only: both GEval metrics are reported but never fail a test case
    # (eval_generate.py keeps its 0.7 thresholds)
    assumptions_metric.threshold = ASSUMPTIONS_THRESHOLD
    refusal_metric.threshold = REFUSAL_THRESHOLD
    if answerable_cases:
        evaluate(test_cases=answerable_cases, metrics=[
            RecordMetric("Final Accuracy", "final"),
            RecordMetric("Schema Recall", "schema_recall"),
            assumptions_metric,
        ], cache_config=no_cache)
    if unanswerable_cases:
        evaluate(test_cases=unanswerable_cases, metrics=[
            RecordMetric("Final Accuracy", "final"),
            RecordMetric("Schema Recall", "schema_recall"),
            refusal_metric,
        ], cache_config=no_cache)
    return records


# ---------------------------------------------------------------- 3. summary

def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else 0.0


def print_summary(records: list[dict]) -> None:
    ans = [r for r in records if r["expected_answerable"]]
    una = [r for r in records if not r["expected_answerable"]]
    correct = lambda r: r["scores"]["final"][0] == 1
    line = "=" * 80
    print(f"\n{line}\nFULL SQL LANE ({len(records)} questions, up to 3 attempts)\n{line}")

    print(f"Final accuracy:        {_pct(sum(map(correct, records)), len(records))}")
    print(f"  answerable correct   {_pct(sum(map(correct, ans)), len(ans))}")
    print(f"  unanswerable refused {_pct(sum(map(correct, una)), len(una))}")

    first_ok = [r for r in ans if correct(r) and r["attempts"] == 1 and not r["errors"]]
    failed_first = [r for r in ans if r["errors"]]
    recovered = [r for r in failed_first if correct(r)]
    print(f"First-attempt correct: {_pct(len(first_ok), len(ans))}")
    print(f"Retry recovery:        {_pct(len(recovered), len(failed_first))}   "
          f"(fixed by a retry / failed on attempt 1)")
    print(f"Schema recall:         {_pct(sum(r['scores']['schema_recall'][0] == 1 for r in records), len(records))} "
          f"all tables")
    print("(Assumptions Quality and Refusal Quality: see the DeepEval results above.)")

    # where the answerable misses ended
    wrong = [r for r in ans if not correct(r)]
    print(f"\nFailure breakdown ({len(wrong)} answerable not correct):")
    last_stage = Counter(r["errors"][-1]["stage"] for r in wrong
                         if r["status"] == "failed" and r["errors"])
    print("  failed, last error at: "
          + (", ".join(f"{s} {n}" for s, n in last_stage.most_common()) or "none"))
    print(f"  ok but WRONG result:   {sum(r['status'] == 'ok' for r in wrong)}"
          f"   <- passed every check yet wrong: the key failure to track")
    print(f"  wrongly unanswerable:  {sum(r['status'] == 'unanswerable' for r in wrong)}")
    print(f"  crashed:               {sum(r['status'] == 'error' for r in wrong)}")
    not_refused = [r for r in una if not correct(r)]
    if not_refused:
        print(f"  unanswerable answered: {len(not_refused)} "
              f"({', '.join(f'{r['id']} status {r['status']}' for r in not_refused)})")

    attempts = Counter(r["attempts"] for r in ans)
    print("\nAttempts used (answerable): "
          + ", ".join(f"{k} -> {attempts[k]}" for k in sorted(attempts)))
    secs = [r["seconds"] for r in records]
    print(f"Latency per question:  median {_percentile(secs, 0.5):.1f}s   "
          f"p95 {_percentile(secs, 0.95):.1f}s   (tokens and cost: see LangSmith)")

    print("\nFinal accuracy by difficulty:")
    for level in ("easy", "medium", "hard"):
        rs = [r for r in records if r["difficulty"] == level]
        if rs:
            print(f"  {level:7} {_pct(sum(map(correct, rs)), len(rs))}")

    print("\nFinal accuracy by rule tested:")
    by_rule = defaultdict(list)
    for r in records:
        for rule in r["rules"]:
            by_rule[rule].append(r)
    for rule in sorted(by_rule):
        rs = by_rule[rule]
        print(f"  {rule:24} {_pct(sum(map(correct, rs)), len(rs))}")

    misses = [r for r in records if not correct(r)]
    print(f"\nMisses ({len(misses)}):" if misses else "\nMisses: none")
    for r in misses:
        print(f"\n  {r['id']} [{r['difficulty']}] {r['question']}")
        print(f"     path: {r['path']}   ({r['attempts']} attempt(s), {r['seconds']:.1f}s)")
        print(f"     {r['scores']['final'][1]}")
        if r["scores"]["schema_recall"][0] < 1:
            print(f"     schema select {r['scores']['schema_recall'][1]}; chose {', '.join(r['tables'])}")
        for i, (sql, err) in enumerate(zip(r["sqls"], r["errors"]), 1):
            print(f"     attempt {i} failed at {err['stage']}: {err['message'][:200]}")
        if r["sqls"]:
            print("     final SQL: " + r["sqls"][-1].replace("\n", "\n                "))


def run_local() -> list[dict]:
    """Standalone convenience: build the lane (default models, shared connection), then run."""
    return run(build_sql_lane())


if __name__ == "__main__":
    print_summary(run_local())
