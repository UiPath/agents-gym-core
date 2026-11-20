"""Builds the HoVer eval-set splits of GEPA (https://arxiv.org/abs/2507.19457).

Source: the HoVer train release (hover_train_release_v1.1.json) from the hover-nlp/hover
GitHub repository at a pinned commit. As in gepa_artifact/benchmarks/hover/hover_data.py and
benchmark.py: keep 3-hop claims, shuffle with Random(seed=0), slice into test=[:40%],
val=[40%:80%], train=[80%:], then trim with Random(seed=1).sample to 150 train, 300 val and
300 test.

Writes evaluations/eval-sets/evaluation-set-hover_{train,val,test}.json and the retrieval
evaluator under evaluations/evaluators/, and builds the BM25 index shared with HotpotQA.
"""

import importlib.util
import json
import os
import random
import urllib.request
import uuid
from pathlib import Path

EVALUATOR_ID = "hover_RetrievalAccuracyEvaluator"

EVALUATOR_JSON = {
    "version": "1.0",
    "id": EVALUATOR_ID,
    "description": (
        "Binary retrieval accuracy: score=1 if all gold document titles are "
        "found in the retrieved passages, 0 otherwise. "
        "Replicates gepa_artifact.benchmarks.hover.hover_utils.discrete_retrieval_eval."
    ),
    "evaluatorTypeId": EVALUATOR_ID,
    "evaluatorSchema": f"file://hover_evaluators.py:{EVALUATOR_ID}",
    "evaluatorConfig": {
        "name": "RetrievalAccuracyEvaluator",
        "description": "",
        "defaultEvaluationCriteria": None,
    },
}

# Written to evaluations/evaluators/custom/hover_evaluators.py; GEPA's discrete_retrieval_eval
# (gepa_artifact/benchmarks/hover/hover_utils.py): 1.0 if every gold title was retrieved.
HOVER_EVALUATOR_PY = '''"""Custom deterministic evaluator for the HoVer multi-hop retrieval agent.

Implements the discrete retrieval accuracy metric from the GEPA paper:
    https://arxiv.org/abs/2507.19457

Score = 1.0 iff all gold document titles are a subset of the titles found
in the retrieved passages.  Titles are normalised via dspy.evaluate.normalize_text
(lower-case, strip punctuation/articles) before comparison.

This exactly replicates:
    gepa_artifact.benchmarks.hover.hover_utils.discrete_retrieval_eval
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


def _normalize_text(text: str) -> str:
    """Normalise a title string as dspy.evaluate.normalize_text does."""
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\\b(a|an|the)\\b", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


def _extract_title(passage: str) -> str:
    """Extract the title from a 'Title | passage text' string."""
    return passage.split(" | ")[0]


class RetrievalAccuracyCriteria(BaseEvaluationCriteria):
    expected_output: dict | list | None = Field(default=None)


class RetrievalAccuracyConfig(BaseEvaluatorConfig[RetrievalAccuracyCriteria]):
    name: str = "RetrievalAccuracyEvaluator"


class RetrievalAccuracyJustification(BaseEvaluatorJustification):
    gold_titles: list[str]
    found_titles: list[str]
    missing_titles: list[str]


class hover_RetrievalAccuracyEvaluator(
    BaseEvaluator[RetrievalAccuracyCriteria, RetrievalAccuracyConfig, str]
):
    """Binary retrieval accuracy: 1.0 iff all gold titles are retrieved."""

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: RetrievalAccuracyCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if isinstance(agent_output, dict):
            retrieved = agent_output.get("retrieved_docs", [])
        else:
            retrieved = agent_output or []

        found_titles = set(
            _normalize_text(_extract_title(doc))
            for doc in retrieved
            if isinstance(doc, str)
        )

        expected = evaluation_criteria.expected_output
        if isinstance(expected, dict):
            gold_raw = expected.get("gold_titles", [])
        elif isinstance(expected, list):
            gold_raw = expected
        else:
            gold_raw = []

        gold_titles = set(_normalize_text(t) for t in gold_raw if t)

        missing = gold_titles - found_titles
        score = 1.0 if not missing else 0.0

        return NumericEvaluationResult(
            score=score,
            details=RetrievalAccuracyJustification(
                expected=str(sorted(gold_titles)),
                actual=str(sorted(found_titles)),
                gold_titles=sorted(gold_titles),
                found_titles=sorted(found_titles),
                missing_titles=sorted(missing),
            ),
        )

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "hover_RetrievalAccuracyEvaluator"
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
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        gold_titles = sorted({fact["key"] for fact in supporting_facts})

        item_name = f"{label}-{i:03d}"
        expected_behavior = f"Retrieve all {len(gold_titles)} supporting documents: {gold_titles}"

        evaluations.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{eval_set_name}/{item_name}")),
                "name": item_name,
                "inputs": {"claim": claim},
                "expectedOutput": {
                    "gold_titles": gold_titles,
                },
                "expectedAgentBehavior": expected_behavior,
                "evaluationCriterias": {
                    EVALUATOR_ID: {
                        "expectedOutput": {"gold_titles": gold_titles},
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
    hover_json_url = (
        "https://raw.githubusercontent.com/hover-nlp/hover/39b84697f196308f398a251a7aea9b82ae0f0562"
        "/data/hover/hover_train_release_v1.1.json"
    )

    seed: int = params.get("seed", 0)

    output_dir = agent_dir / "evaluations" / "eval-sets"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluators_dir = agent_dir / "evaluations" / "evaluators"
    evaluators_dir.mkdir(parents=True, exist_ok=True)
    custom_dir = evaluators_dir / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    evaluator_path = evaluators_dir / f"evaluator-{EVALUATOR_ID}.json"
    evaluator_path.write_text(json.dumps(EVALUATOR_JSON, indent=2) + "\n")
    print(f"Wrote evaluator -> {evaluator_path}")

    custom_py = custom_dir / "hover_evaluators.py"
    custom_py.write_text(HOVER_EVALUATOR_PY)
    print(f"Wrote evaluator -> {custom_py}")

    # Use the release JSON underlying GEPA's Hugging Face dataset to preserve its source
    # rows across datasets library versions.
    print(f"Downloading HoVer train JSON from {hover_json_url} ...")
    with urllib.request.urlopen(hover_json_url) as response:  # noqa: S310
        raw_data: list[dict] = json.loads(response.read().decode())
    print(f"  {len(raw_data)} total examples")

    # supporting_facts: the release has [[title, sent_id], ...]; GEPA's code reads
    # [{"key": title}, ...].
    three_hop = []
    for example in raw_data:
        # GEPA filters on 3 unique supporting documents; on this release both select the same
        # 6,084 claims in the same order, so the splits are GEPA's.
        if example.get("num_hops") == 3:
            supporting_facts = [{"key": sf[0]} for sf in example["supporting_facts"]]
            three_hop.append(
                {
                    "claim": example["claim"],
                    "supporting_facts": supporting_facts,
                    "label": example["label"],
                }
            )
    print(f"  {len(three_hop)} 3-hop examples")

    shuffled = list(three_hop)
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_set_raw = shuffled[: int(0.4 * total)]
    val_set_raw = shuffled[int(0.4 * total) : int(0.8 * total)]
    train_set_raw = shuffled[int(0.8 * total) :]
    print(
        f"  Raw splits: {len(train_set_raw)} train, "
        f"{len(val_set_raw)} val, {len(test_set_raw)} test"
    )

    train_set = _trim_dataset(train_set_raw, 150, seed=1)
    val_set = _trim_dataset(val_set_raw, 300, seed=1)
    test_set = _trim_dataset(test_set_raw, 300, seed=1)
    print(f"  After trim: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    splits = [
        (train_set, "evaluation-set-hover_train", "train"),
        (val_set, "evaluation-set-hover_val", "val"),
        (test_set, "evaluation-set-hover_test", "test"),
    ]
    for examples, eval_set_name, label in splits:
        eval_set = _build_eval_set(examples, eval_set_name=eval_set_name, label=label)
        path = output_dir / f"{eval_set_name}.json"
        path.write_text(json.dumps(eval_set, indent=2) + "\n")
        print(f"Wrote eval-set  -> {path}  ({len(examples)} datapoints)")

    _build_bm25_index()


def _build_bm25_index() -> None:
    """Build GEPA's BM25 index next to hover_program.py in the submodule, if missing.

    init_retriever() downloads a 580 MB archive into the working directory, so it runs from
    the benchmark directory; the archive is deleted once the index is built.
    """
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
