"""Builds the HotpotQA eval-set splits of GEPA (https://arxiv.org/abs/2507.19457).

Source: the Hugging Face dataset ``hotpotqa/hotpot_qa``, fullwiki config, train split (90,447
examples), at a pinned revision. As in gepa_artifact/benchmarks/benchmark.py and
hotpotQA/hotpot_data.py: no filtering and no shuffle; slice into test=[:40%], val=[40%:80%],
train=[80%:], then trim with Random(seed=1).sample to 150 train, 300 val and 300 test.

Writes evaluations/eval-sets/evaluation-set-hotpotqa_{train,val,test}.json and the
exact-match evaluator under evaluations/evaluators/, and builds the BM25 index shared with
HoVer.
"""

import importlib.util
import json
import os
import random
import uuid
from pathlib import Path

# Pinned so the splits stay byte-identical.
HOTPOT_QA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"

EVALUATOR_ID = "hotpotqa_AnswerExactMatchEvaluator"

EVALUATOR_JSON = {
    "version": "1.0",
    "id": EVALUATOR_ID,
    "description": (
        "Normalised exact-match evaluator for HotpotQA: score=1 if the "
        "predicted answer matches the gold answer after Unicode NFD "
        "normalisation, lowercasing, punctuation removal, English article "
        "removal and whitespace collapse. "
        "Replicates dspy.evaluate.answer_exact_match."
    ),
    "evaluatorTypeId": EVALUATOR_ID,
    "evaluatorSchema": f"file://hotpotqa_evaluators.py:{EVALUATOR_ID}",
    "evaluatorConfig": {
        "name": "AnswerExactMatchEvaluator",
        "description": "",
        "defaultEvaluationCriteria": None,
    },
}

# Written to evaluations/evaluators/custom/hotpotqa_evaluators.py; dspy's answer_exact_match.
HOTPOTQA_EVALUATOR_PY = '''"""Custom deterministic evaluator for the HotpotQA multi-hop QA agent.

Implements the normalised exact-match metric used in the GEPA paper:
    https://arxiv.org/abs/2507.19457

Score = 1.0 iff normalize_text(predicted_answer) == normalize_text(gold_answer).

The normalisation pipeline (matching dspy.evaluate.normalize_text):
  1. Unicode NFD normalisation
  2. Lowercasing
  3. Punctuation removal
  4. English article removal ("a", "an", "the")
  5. Whitespace collapse
"""

import re
import string
import unicodedata

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import WorkloadExecution, EvaluationResult, NumericEvaluationResult


def _normalize_text(s: str) -> str:
    """Normalise text — mirrors dspy.evaluate.normalize_text exactly."""
    s = unicodedata.normalize("NFD", s)
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\\b(a|an|the)\\b", " ", s)
    return " ".join(s.split())


class AnswerExactMatchCriteria(BaseEvaluationCriteria):
    expected_output: dict | str | None = Field(default=None)


class AnswerExactMatchConfig(BaseEvaluatorConfig[AnswerExactMatchCriteria]):
    name: str = "AnswerExactMatchEvaluator"


class AnswerExactMatchJustification(BaseEvaluatorJustification):
    normalized_predicted: str
    normalized_expected: str


class hotpotqa_AnswerExactMatchEvaluator(
    BaseEvaluator[AnswerExactMatchCriteria, AnswerExactMatchConfig, str]
):
    """Normalised exact-match: 1.0 iff answers match after normalisation."""

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: AnswerExactMatchCriteria,
    ) -> EvaluationResult:
        # Extract predicted answer
        agent_output = workload_execution.workload_output
        if isinstance(agent_output, dict):
            predicted = agent_output.get("answer", "")
        elif isinstance(agent_output, str):
            predicted = agent_output
        else:
            predicted = ""

        # Extract expected answer
        expected = evaluation_criteria.expected_output
        if isinstance(expected, dict):
            gold = expected.get("answer", "")
        elif isinstance(expected, str):
            gold = expected
        else:
            gold = ""

        norm_pred = _normalize_text(str(predicted))
        norm_gold = _normalize_text(str(gold))
        score = 1.0 if norm_pred == norm_gold else 0.0

        return NumericEvaluationResult(
            score=score,
            details=AnswerExactMatchJustification(
                expected=gold,
                actual=predicted,
                normalized_predicted=norm_pred,
                normalized_expected=norm_gold,
            ),
        )

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "hotpotqa_AnswerExactMatchEvaluator"
'''


def _trim_dataset(dataset: list, size: int, seed: int = 1) -> list:
    """Subsample to *size* with Random(seed).sample, as GEPA's benchmark.trim_dataset."""
    if size is None or size >= len(dataset):
        return dataset
    rng = random.Random()
    rng.seed(seed)
    return rng.sample(dataset, size)


def _build_eval_set(
    examples: list[dict],
    eval_set_name: str,
    label: str,
) -> dict:
    eval_set_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, eval_set_name))
    evaluations = []
    for i, example in enumerate(examples):
        question = example["question"]
        answer = example["answer"]

        item_name = f"{label}-{i:03d}"
        expected_behavior = f"Answer the multi-hop question correctly. Expected: {answer}"

        evaluations.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{eval_set_name}/{item_name}")),
                "name": item_name,
                "inputs": {"question": question},
                "expectedOutput": {
                    "answer": answer,
                },
                "expectedAgentBehavior": expected_behavior,
                "evaluationCriterias": {
                    EVALUATOR_ID: {
                        "expectedOutput": {"answer": answer},
                        "expectedAgentBehavior": expected_behavior,
                    },
                },
            }
        )

    return {
        "version": "1.0",
        "id": eval_set_id,
        "name": eval_set_name,
        "evaluatorRefs": [EVALUATOR_ID],
        "evaluations": evaluations,
    }


def init(agent_dir: Path, params: dict) -> None:
    """Write the evaluator and the three splits, then build the BM25 index."""
    from datasets import load_dataset

    output_dir = agent_dir / "evaluations" / "eval-sets"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluators_dir = agent_dir / "evaluations" / "evaluators"
    evaluators_dir.mkdir(parents=True, exist_ok=True)
    custom_dir = evaluators_dir / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    evaluator_path = evaluators_dir / f"evaluator-{EVALUATOR_ID}.json"
    evaluator_path.write_text(json.dumps(EVALUATOR_JSON, indent=2) + "\n")
    print(f"Wrote evaluator -> {evaluator_path}")

    custom_py = custom_dir / "hotpotqa_evaluators.py"
    custom_py.write_text(HOTPOTQA_EVALUATOR_PY)
    print(f"Wrote evaluator -> {custom_py}")

    # The dataset gepa_artifact/benchmarks/hotpotQA/hotpot_data.py loads as "hotpot_qa".
    print("Loading HotpotQA fullwiki train split from HuggingFace...")
    raw_dataset = load_dataset(
        "hotpotqa/hotpot_qa", "fullwiki", split="train", revision=HOTPOT_QA_REVISION
    )
    all_examples = [dict(x) for x in raw_dataset]
    print(f"  {len(all_examples)} total examples")

    # benchmark.py's create_splits slices without shuffling.
    total = len(all_examples)
    test_set_raw = all_examples[: int(0.4 * total)]
    val_set_raw = all_examples[int(0.4 * total) : int(0.8 * total)]
    train_set_raw = all_examples[int(0.8 * total) :]
    print(
        f"  Raw splits: {len(train_set_raw)} train, "
        f"{len(val_set_raw)} val, {len(test_set_raw)} test"
    )

    train_set = _trim_dataset(train_set_raw, 150, seed=1)
    val_set = _trim_dataset(val_set_raw, 300, seed=1)
    test_set = _trim_dataset(test_set_raw, 300, seed=1)
    print(f"  After trim: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    splits = [
        (train_set, "evaluation-set-hotpotqa_train", "train"),
        (val_set, "evaluation-set-hotpotqa_val", "val"),
        (test_set, "evaluation-set-hotpotqa_test", "test"),
    ]
    for examples, eval_set_name, label in splits:
        eval_set = _build_eval_set(examples, eval_set_name=eval_set_name, label=label)
        path = output_dir / f"{eval_set_name}.json"
        path.write_text(json.dumps(eval_set, indent=2) + "\n")
        print(f"Wrote eval-set  -> {path}  ({len(examples)} datapoints)")

    # GEPA's HotpotQA program imports `search` from hover_program.py (same BM25 index).
    _build_bm25_index()


def _build_bm25_index() -> None:
    """Build the shared BM25 retrieval index if it doesn't already exist."""
    print("  Checking BM25 retrieval index...")

    from gepa_artifact.benchmarks.hover.hover_program import init_retriever

    from agents_gym_core.gepa_artifact import verified_wiki_abstracts

    spec = importlib.util.find_spec("gepa_artifact.benchmarks.hover.hover_program")
    assert spec and spec.origin
    benchmark_dir = Path(spec.origin).parent
    index_path = benchmark_dir / "bm25s_retriever"
    tar_path = benchmark_dir / "wiki.abstracts.2017.tar.gz"

    if index_path.exists():
        print(f"  BM25 index already exists at {index_path} — skipping.")
    else:
        print("  Building BM25 retrieval index from the gepa-artifact submodule...")
        original_cwd = os.getcwd()
        try:
            os.chdir(benchmark_dir)
            with verified_wiki_abstracts():
                init_retriever()
        finally:
            os.chdir(original_cwd)
        if tar_path.exists():
            tar_path.unlink()
            print(f"  Deleted {tar_path.name} (not needed after index build)")
        print(f"  BM25 index ready at {index_path}")
