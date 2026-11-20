"""Builds the LiveBench-Math eval sets of GEPA from the Hugging Face dataset livebench/math.

As in gepa_artifact/benchmarks/livebench_math/livebenchmath_data.py: shuffle with
Random(seed=0) and cut at 33% and 66% into train, val and test.
"""

import json
import random
import uuid
from datetime import date, datetime
from pathlib import Path

from agents_gym_core.utils.benchmark_demos import write_demos

# Pinned so the splits stay byte-identical.
LIVEBENCH_MATH_REVISION = "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb"


def _json_default(obj):  # type: ignore
    """Serialize the dataset's date fields."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


EVALUATOR_ID = "livebench_math_LiveBenchMathEvaluator"

EVALUATOR_JSON = {
    "version": "1.0",
    "id": EVALUATOR_ID,
    "description": "LiveBench-Math evaluator using competition-specific scoring (AMC/AIME/IMO/AMPS)",
    "evaluatorTypeId": "LiveBenchMathEvaluator",
    "evaluatorSchema": "file://livebench_math_evaluator.py:LiveBenchMathEvaluator",
    "evaluatorConfig": {
        "name": "LiveBenchMathEvaluator",
        "description": "Exact-match math scoring across competition formats",
        "defaultEvaluationCriteria": None,
    },
}


EVALUATOR_PY = '''\
"""LiveBench-Math evaluator — routes to competition-specific scoring from the GEPA artifact."""

import asyncio
import builtins as _builtins
import concurrent.futures as _cf
import importlib as _importlib
import threading
from typing import Any

from pydantic import ConfigDict, Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
)
from uipath.eval.models import WorkloadExecution, EvaluationResult, NumericEvaluationResult

# gepa_artifact is importable: agents_gym_core puts the checkout on sys.path at import.
from gepa_artifact.benchmarks.livebench_math.livebenchmath_utils.metric import (
    calculate_livebench_score,
)

import multiprocessing as _mp

_amps_utils = _importlib.import_module(
    "gepa_artifact.benchmarks.livebench_math.livebenchmath_utils.AMPS_Hard.utils"
)

_fork_ctx = _mp.get_context("fork")


def _run_with_timeout_fork(func, args=(), timeout=8):
    q = _fork_ctx.Queue()

    def wrapper(queue):
        try:
            result = func(*args)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    p = _fork_ctx.Process(target=wrapper, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise _builtins.TimeoutError("Operation timed out")

    # Child may have been killed by a signal (SIGSEGV, OOM, etc.) before
    # putting anything in the queue — detect this and fail fast instead of hanging.
    if p.exitcode is not None and p.exitcode < 0:
        raise RuntimeError(
            f"Worker process killed by signal {-p.exitcode} (native crash or OOM)"
        )

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result


_amps_utils.run_with_timeout = _run_with_timeout_fork

# The vendored scorer is not thread-safe, so answers are scored one at a time.
_SCORER_LOCK = threading.Lock()


def _score(question_d: dict, answer: str) -> tuple[Any, Any]:
    with _SCORER_LOCK:
        return calculate_livebench_score(question_d, answer, debug=False)


class LiveBenchMathCriteria(BaseEvaluationCriteria):
    model_config = ConfigDict(populate_by_name=True)

    expectedOutput: dict[str, Any] = Field(alias="expectedOutput")
    expectedAgentBehavior: str = Field(default="", alias="expectedAgentBehavior")


class LiveBenchMathConfig(BaseEvaluatorConfig[LiveBenchMathCriteria]):
    name: str = "LiveBenchMathEvaluator"


class LiveBenchMathEvaluator(
    BaseEvaluator[LiveBenchMathCriteria, LiveBenchMathConfig, str]
):
    @classmethod
    def get_evaluator_id(cls) -> str:
        return "LiveBenchMathEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: LiveBenchMathCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if isinstance(agent_output, str):
            answer = agent_output
        elif isinstance(agent_output, dict):
            answer = agent_output.get("answer", "")
        else:
            return NumericEvaluationResult(score=0.0, details="No agent output")

        expected = evaluation_criteria.expectedOutput
        question_d = expected.get("question_d", {})

        if not answer or not question_d:
            return NumericEvaluationResult(score=0.0, details="No answer or no question data")

        try:
            # Off the event loop: one answer can take seconds to score.
            score, feedback = await asyncio.to_thread(_score, question_d, answer)
        except Exception as e:
            return NumericEvaluationResult(score=0.0, details=f"Scoring error: {e}")

        return NumericEvaluationResult(score=float(score), details=feedback or "")
'''


def init(agent_dir: Path, params: dict) -> None:
    """Generate LiveBench-Math evaluation sets and custom evaluator."""
    seed = params.get("seed", 0)

    from datasets import load_dataset

    print("Loading livebench/math from HuggingFace...")
    raw = load_dataset("livebench/math", revision=LIVEBENCH_MATH_REVISION)
    all_examples = list(raw["test"])
    total = len(all_examples)

    random.Random(seed).shuffle(all_examples)

    train_end = int(total * 0.33)
    val_end = int(total * 0.66)
    train_examples = all_examples[:train_end]
    val_examples = all_examples[train_end:val_end]
    test_examples = all_examples[val_end:]

    # GEPA's Benchmark trim: at most 150 train, 300 val, 300 test, Random(1).sample.
    rng = random.Random(1)
    if len(train_examples) > 150:
        train_examples = rng.sample(train_examples, 150)
    if len(val_examples) > 300:
        val_examples = rng.sample(val_examples, 300)
    if len(test_examples) > 300:
        test_examples = rng.sample(test_examples, 300)

    evaluator_dir = agent_dir / "evaluations" / "evaluators"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    (evaluator_dir / "custom").mkdir(exist_ok=True)

    with open(evaluator_dir / f"evaluator-{EVALUATOR_ID}.json", "w") as f:
        json.dump(EVALUATOR_JSON, f, indent=2)

    custom_dir = evaluator_dir / "custom"
    custom_dir.mkdir(exist_ok=True)
    custom_py = custom_dir / "livebench_math_evaluator.py"
    custom_py.write_text(EVALUATOR_PY)
    print(f"Wrote evaluator -> {custom_py}")

    eval_sets_dir = agent_dir / "evaluations" / "eval-sets"
    eval_sets_dir.mkdir(parents=True, exist_ok=True)

    def _make_eval_set(examples: list[dict], name: str) -> dict:
        evaluations = []
        for ex in examples:
            qid = ex.get("question_id", str(uuid.uuid4())[:8])
            question = ex["turns"][0]
            ground_truth = ex.get("ground_truth", "")
            task = ex.get("task", "unknown")
            subtask = ex.get("subtask", task)

            evaluations.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"livebench-math-{qid}")),
                    "name": f"livebench-math-{qid}",
                    "inputs": {"question": question},
                    "expectedOutput": {
                        "ground_truth": ground_truth,
                        "question_d": ex,
                    },
                    "expectedAgentBehavior": f"Solve the {subtask} math problem correctly",
                    "evaluationCriterias": {
                        EVALUATOR_ID: {
                            "expectedOutput": {
                                "ground_truth": ground_truth,
                                "question_d": ex,
                            },
                            "expectedAgentBehavior": f"Solve the {subtask} math problem correctly",
                        }
                    },
                }
            )

        return {
            "version": "1.0",
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"livebench-math-evalset-{name}")),
            "name": f"evaluation-set-livebench-math-{name}"
            if name != "all"
            else "evaluation-set-livebench-math",
            "evaluatorRefs": [EVALUATOR_ID],
            "evaluations": evaluations,
        }

    # The base version's eval set is the test split.
    eval_set = _make_eval_set(test_examples, "all")  # type: ignore
    with open(eval_sets_dir / "evaluation-set-livebench-math.json", "w") as f:
        json.dump(eval_set, f, indent=2, default=_json_default)

    for split_name, split_data in [
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ]:
        split_set = _make_eval_set(split_data, split_name)  # type: ignore
        with open(eval_sets_dir / f"evaluation-set-livebench-math-{split_name}.json", "w") as f:
            json.dump(split_set, f, indent=2, default=_json_default)

    for refs in params.get("demo_refs", []):
        print(f"Wrote demos -> {write_demos(agent_dir, refs)}")

    print("LiveBench-Math init complete:")
    print(f"  Total HF examples: {total}")
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
    print(f"  Evaluator: {EVALUATOR_ID}")
