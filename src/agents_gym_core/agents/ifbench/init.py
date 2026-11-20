"""Builds the IFBench eval sets of GEPA from the JSONL files in the gepa-artifact submodule.

As in gepa_artifact/benchmarks/IFBench/ifbench_data.py: test is IFBench_test.jsonl; val and
train are rows [:300] and [300:600] of IFBench_train.jsonl.
"""

import json
import uuid
from pathlib import Path

EVALUATOR_ID = "ifbench_IFBenchInstructionFollowingEvaluator"

EVALUATOR_JSON = {
    "version": "1.0",
    "id": EVALUATOR_ID,
    "description": "Instruction-following evaluator using IFBench metric (81 instruction types)",
    "evaluatorTypeId": "IFBenchInstructionFollowingEvaluator",
    "evaluatorSchema": "file://ifbench_instruction_following_evaluator.py:IFBenchInstructionFollowingEvaluator",
    "evaluatorConfig": {
        "name": "IFBenchInstructionFollowingEvaluator",
        "description": "Checks instruction-following rate across 8 lenient response variants",
        "defaultEvaluationCriteria": None,
    },
}

EVALUATOR_PY = '''\
"""IFBench instruction-following evaluator — wraps the GEPA metric directly."""

from typing import Any

import dspy
import nltk as _nltk
from pydantic import ConfigDict, Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
)
from uipath.eval.models import WorkloadExecution, EvaluationResult, NumericEvaluationResult

from agents_gym_core.gepa_artifact import (
    ensure_gepa_artifact_importable,
    missing_ifbench_nltk_data,
    skip_spacy_model_download,
)

if _missing := missing_ifbench_nltk_data():
    raise RuntimeError(f"IFBench's scorer needs NLTK data {_missing}: run `gym init ifbench`")

# IFBench's scorer calls nltk.download on every call. `gym init ifbench` installs
# the data, so the call is a no-op here.
_nltk.download = lambda *a, **kw: True
# Same for spaCy's model download (it shells out to pip, absent in uv environments).
skip_spacy_model_download()

ensure_gepa_artifact_importable()

from gepa_artifact.benchmarks.IFBench.ifbench_metric import metric_with_feedback  # noqa: E402


class IFBenchCriteria(BaseEvaluationCriteria):
    model_config = ConfigDict(populate_by_name=True)

    expectedOutput: dict[str, Any] = Field(alias="expectedOutput")
    expectedAgentBehavior: str = Field(default="", alias="expectedAgentBehavior")


class IFBenchConfig(BaseEvaluatorConfig[IFBenchCriteria]):
    name: str = "IFBenchInstructionFollowingEvaluator"


class IFBenchInstructionFollowingEvaluator(
    BaseEvaluator[IFBenchCriteria, IFBenchConfig, str]
):
    @classmethod
    def get_evaluator_id(cls) -> str:
        return "IFBenchInstructionFollowingEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: IFBenchCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if isinstance(agent_output, str):
            response = agent_output
        elif isinstance(agent_output, dict):
            response = agent_output.get("response", "")
        else:
            return NumericEvaluationResult(score=0.0, details="No agent output")

        expected = evaluation_criteria.expectedOutput
        instruction_id_list = expected.get("instruction_id_list", [])
        kwargs_list = expected.get("kwargs", [])
        prompt = expected.get("prompt", "")

        if not response or not instruction_id_list:
            return NumericEvaluationResult(score=0.0, details="No response or no instructions")

        example = dspy.Example(
            prompt=prompt,
            instruction_id_list=instruction_id_list,
            kwargs=kwargs_list,
        ).with_inputs("prompt")
        pred = dspy.Prediction(response=response)

        result = metric_with_feedback(example, pred)

        return NumericEvaluationResult(
            score=float(result.score),
            details=result.feedback or "",
        )
'''


def init(agent_dir: Path, params: dict) -> None:
    """Generate IFBench evaluation sets and custom evaluator."""
    num_datapoints = params.get("num_datapoints", 294)
    params.get("seed", 0)

    from nltk.downloader import Downloader  # not nltk.download, which the evaluator stubs out

    from agents_gym_core.gepa_artifact import gepa_artifact_root, missing_ifbench_nltk_data

    for name in missing_ifbench_nltk_data():
        Downloader().download(name, quiet=True)
    if missing := missing_ifbench_nltk_data():
        raise RuntimeError(f"could not download the NLTK data IFBench's scorer needs: {missing}")

    gepa_data = gepa_artifact_root() / "gepa_artifact" / "benchmarks" / "IFBench" / "data"

    test_file = gepa_data / "IFBench_test.jsonl"
    train_file = gepa_data / "IFBench_train.jsonl"

    if not test_file.exists():
        raise RuntimeError(
            f"IFBench data not found at {gepa_data}: check out the gepa-artifact submodule "
            "(git submodule update --init gepa-artifact)"
        )

    test_examples = []
    with open(test_file) as f:
        for line in f:
            test_examples.append(json.loads(line))

    all_train = []
    with open(train_file) as f:
        for line in f:
            all_train.append(json.loads(line))

    val_examples = all_train[:300]
    train_examples = all_train[300:600]

    evaluator_dir = agent_dir / "evaluations" / "evaluators"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    custom_dir = evaluator_dir / "custom"
    custom_dir.mkdir(exist_ok=True)

    with open(evaluator_dir / f"evaluator-{EVALUATOR_ID}.json", "w") as f:
        json.dump(EVALUATOR_JSON, f, indent=2)

    custom_py = custom_dir / "ifbench_instruction_following_evaluator.py"
    custom_py.write_text(EVALUATOR_PY)
    print(f"Wrote evaluator -> {custom_py}")

    eval_sets_dir = agent_dir / "evaluations" / "eval-sets"
    eval_sets_dir.mkdir(parents=True, exist_ok=True)

    def _make_eval_set(examples: list[dict], name: str, max_n: int | None = None) -> dict:
        if max_n and len(examples) > max_n:
            import random

            # As GEPA's Benchmark.trim_dataset.
            rng = random.Random(1)
            examples = rng.sample(examples, max_n)

        evaluations = []
        for ex in examples:
            key = str(ex.get("key", ""))
            prompt = ex["prompt"]
            instruction_id_list = ex.get("instruction_id_list", ex.get("instruction_id", []))
            kwargs = ex.get("kwargs", [])

            evaluations.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ifbench-{key}")),
                    "name": f"ifbench-{key}",
                    "inputs": {"prompt": prompt},
                    "expectedOutput": {
                        "instruction_id_list": instruction_id_list,
                        "kwargs": kwargs,
                        "prompt": prompt,
                    },
                    "expectedAgentBehavior": f"Follow all {len(instruction_id_list)} instruction(s) in the prompt",
                    "evaluationCriterias": {
                        EVALUATOR_ID: {
                            "expectedOutput": {
                                "instruction_id_list": instruction_id_list,
                                "kwargs": kwargs,
                                "prompt": prompt,
                            },
                            "expectedAgentBehavior": f"Follow all {len(instruction_id_list)} instruction(s)",
                        }
                    },
                }
            )

        return {
            "version": "1.0",
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ifbench-evalset-{name}")),
            "name": f"evaluation-set-ifbench-{name}" if name != "all" else "evaluation-set-ifbench",
            "evaluatorRefs": [EVALUATOR_ID],
            "evaluations": evaluations,
        }

    # The base version's eval set is the test split.
    eval_set = _make_eval_set(test_examples, "all", max_n=num_datapoints)
    with open(eval_sets_dir / "evaluation-set-ifbench.json", "w") as f:
        json.dump(eval_set, f, indent=2)

    for split_name, split_data, max_n in [
        ("train", train_examples, 150),
        ("val", val_examples, 300),
        ("test", test_examples, 294),
    ]:
        split_set = _make_eval_set(split_data, split_name, max_n=max_n)
        with open(eval_sets_dir / f"evaluation-set-ifbench-{split_name}.json", "w") as f:
            json.dump(split_set, f, indent=2)

    print("IFBench init complete:")
    print(f"  Test eval set: {len(eval_set['evaluations'])} datapoints")
    print(
        f"  Train: {min(150, len(train_examples))}, Val: {min(300, len(val_examples))}, Test: {min(294, len(test_examples))}"
    )
    print(f"  Evaluator: {EVALUATOR_ID}")
