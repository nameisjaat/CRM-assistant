"""Evaluate schema select: does it pick every table each question needs?

    .venv\\Scripts\\python.exe eval\\eval_schema_select.py

Runs the real schema select node (gpt-4o-mini, temperature 0, as in the lane) on every
question in eval/sql_lane_questions.yaml and scores it with TableRecall:

    recall = needed tables that were selected / needed tables

Extra tables cost nothing (schema select is told an extra table costs little); a missing
table makes the question impossible to answer. A question passes only at recall 1.0.

Cost: one gpt-4o-mini call per question (~3,500 input tokens each). No judge model and no
database: the metric is plain counting. Reads OPENAI_API_KEY from the project's .env.
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))            # so `sql_lane` imports when run as a script

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from sql_lane.schema_select import make_schema_select

load_dotenv(ROOT / ".env")

GOLDEN_PATH = ROOT / "eval" / "sql_lane_questions.yaml"
SCHEMA_MODEL = "gpt-4o-mini"             # the same model the lane uses (sql_lane/graph.py)
THRESHOLD = 0.8                          # pass only if every needed table was selected
ONLY_VERIFIED = True                     # count only entries with verified: true


# ---------------------------------------------------------------- 1. load goldens

def load_goldens(path: Path = GOLDEN_PATH) -> list[dict]:
    goldens = yaml.safe_load(path.read_text(encoding="utf-8"))
    if ONLY_VERIFIED:
        goldens = [g for g in goldens if g.get("verified")]
    return goldens


# ---------------------------------------------------------------- 2. the metric

class TableRecall(BaseMetric):
    """Share of the needed tables that schema select chose. Extra tables are not penalised.

    Reads both lists from test_case.metadata: "expected_tables" and "selected_tables".
    Plain counting, no LLM.
    """

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.evaluation_model = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = set(test_case.metadata["expected_tables"])
        selected = set(test_case.metadata["selected_tables"])
        missing = sorted(expected - selected)
        extra = sorted(selected - expected)

        self.score = len(expected & selected) / len(expected) if expected else 1.0
        self.reason = (f"{len(expected & selected)}/{len(expected)} needed tables selected. "
                       f"Missing: {', '.join(missing) or 'none'}. "
                       f"Extra (not penalised): {', '.join(extra) or 'none'}.")
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Table Recall"


# ---------------------------------------------------------------- 3. run and evaluate

def run(schema_select) -> list[dict]:
    """Run schema select on every golden question; return one record per question."""
    goldens = load_goldens()
    records, test_cases = [], []

    for g in goldens:
        selected = schema_select({"question": g["question"]})["tables"]   # the node's final list
        record = {"id": g["id"], "question": g["question"], "difficulty": g["difficulty"],
                  "answerable": g.get("answerable", True),
                  "expected": list(g["tables"]), "selected": list(selected)}
        records.append(record)
        test_cases.append(LLMTestCase(
            input=g["question"],
            actual_output=", ".join(selected),
            expected_output=", ".join(g["tables"]),
            metadata={"id": g["id"], "difficulty": g["difficulty"],
                      "expected_tables": list(g["tables"]), "selected_tables": list(selected)},
        ))

    # no DeepEval result cache: every run scores fresh, and its .deepeval/ files can't
    # be left locked (OneDrive sync, overlapping async writes)
    evaluate(test_cases=test_cases, metrics=[TableRecall()],
             cache_config=CacheConfig(write_cache=False, use_cache=False))

    # the same score per record, for the summary below (plain counting, no LLM)
    for r in records:
        expected, selected = set(r["expected"]), set(r["selected"])
        r["recall"] = len(expected & selected) / len(expected) if expected else 1.0
        r["missing"] = sorted(expected - selected)
        r["extra"] = sorted(selected - expected)
    return records


# ---------------------------------------------------------------- 4. summary

def print_summary(records: list[dict]) -> None:
    n = len(records)
    passed = [r for r in records if r["recall"] >= THRESHOLD]
    line = "=" * 80
    print(f"\n{line}\nSCHEMA SELECT: TABLE RECALL ({n} questions, model {SCHEMA_MODEL})\n{line}")
    print(f"Mean recall:            {sum(r['recall'] for r in records) / n:.3f}")
    print(f"All needed tables:      {len(passed)}/{n} ({100 * len(passed) / n:.1f}%)")
    print(f"Avg extra tables:       {sum(len(r['extra']) for r in records) / n:.2f}   (not penalised)")

    print("\nBy difficulty:")
    groups = defaultdict(list)
    for r in records:
        groups[r["difficulty"]].append(r)
    for level in ("easy", "medium", "hard"):
        rs = groups.get(level, [])
        if rs:
            ok = sum(r["recall"] >= THRESHOLD for r in rs)
            mean = sum(r["recall"] for r in rs) / len(rs)
            print(f"  {level:7} mean recall {mean:.3f}   all tables {ok}/{len(rs)}")

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    for label, rs in (("Answerable", answerable), ("Unanswerable", unanswerable)):
        if rs:
            ok = sum(r["recall"] >= THRESHOLD for r in rs)
            print(f"{label + ':':16}  all tables {ok}/{len(rs)}")

    misses = [r for r in records if r["recall"] < THRESHOLD]
    print(f"\nMisses ({len(misses)}):" if misses else "\nMisses: none")
    for r in misses:
        print(f"  {r['id']} recall {r['recall']:.2f}  {r['question']}")
        print(f"       missing:  {', '.join(r['missing'])}")
        print(f"       selected: {', '.join(r['selected'])}")


def run_local() -> list[dict]:
    """Standalone convenience: build the schema select node, then run."""
    llm = ChatOpenAI(model=SCHEMA_MODEL, temperature=0)
    return run(make_schema_select(llm))


if __name__ == "__main__":
    print_summary(run_local())
