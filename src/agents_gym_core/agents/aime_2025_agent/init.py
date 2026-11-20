"""Builds the AIME eval sets from Hugging Face datasets at pinned revisions.

- evaluation-set-aime_2025.json       30 problems (MathArena/aime_2025), the test set
- evaluation-set-aime_2022_2024.json  90 problems (AI-MO/aimo-validation-aime)
- evaluation-set-aime_train.json      45 of the 90, after a Random(seed).shuffle
- evaluation-set-aime_val.json        the other 45
"""

import json
import random
from pathlib import Path

from agents_gym_core.utils.benchmark_demos import write_demos

# Pinned so the splits stay byte-identical.
AIME_2025_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"
AIMO_AIME_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"

EVALUATOR = {
    "version": "1.0",
    "id": "aime_2025_ExactMatchEvaluator",
    "description": "Exact match evaluator for the answer field in aime_2025_agent",
    "evaluatorTypeId": "uipath-exact-match",
    "evaluatorConfig": {
        "name": "ExactMatchEvaluator",
        "description": "Evaluates if the answer field exactly matches the expected integer answer",
        "defaultEvaluationCriteria": None,
        "targetOutputKey": "answer",
        "caseSensitive": False,
        "negated": False,
    },
}


def init(agent_dir: Path, params: dict) -> None:
    from datasets import load_dataset

    from agents_gym_core.agents.aime_2025_agent.scripts.build_dataset import build

    output_dir = agent_dir / "evaluations" / "eval-sets"
    seed = params.get("seed", 0)

    evaluators_dir = agent_dir / "evaluations" / "evaluators"
    evaluators_dir.mkdir(parents=True, exist_ok=True)
    evaluator_path = evaluators_dir / f"evaluator-{EVALUATOR['id']}.json"
    evaluator_path.write_text(json.dumps(EVALUATOR, indent=2) + "\n")
    print(f"Wrote evaluator -> {evaluator_path}")

    print("Loading AIME 2025 from MathArena/aime_2025...")
    aime_2025 = load_dataset(
        "MathArena/aime_2025", split="train", revision=AIME_2025_REVISION
    ).to_list()
    print(f"  {len(aime_2025)} problems")
    build(
        aime_2025,
        output_dir / "evaluation-set-aime_2025.json",
        "evaluation-set-aime_2025",
        "aime-2025",
        question_key="problem",
        answer_key="answer",
    )

    print("Loading AIME 2022-2024 from AI-MO/aimo-validation-aime...")
    aime_hist = load_dataset(
        "AI-MO/aimo-validation-aime", split="train", revision=AIMO_AIME_REVISION
    ).to_list()
    print(f"  {len(aime_hist)} problems")
    build(
        aime_hist,
        output_dir / "evaluation-set-aime_2022_2024.json",
        "evaluation-set-aime_2022_2024",
        "aime-2022-2024",
        question_key="problem",
        answer_key="answer",
        solution_key="solution",
    )

    shuffled = list(aime_hist)
    random.Random(seed).shuffle(shuffled)
    mid = len(shuffled) // 2
    train_data, val_data = shuffled[:mid], shuffled[mid:]
    print(f"  Split: {len(train_data)} train, {len(val_data)} val")
    build(
        train_data,
        output_dir / "evaluation-set-aime_train.json",
        "evaluation-set-aime_train",
        "aime-train",
        question_key="problem",
        answer_key="answer",
        solution_key="solution",
    )
    build(
        val_data,
        output_dir / "evaluation-set-aime_val.json",
        "evaluation-set-aime_val",
        "aime-val",
        question_key="problem",
        answer_key="answer",
        solution_key="solution",
    )

    for refs in params.get("demo_refs", []):
        print(f"Wrote demos -> {write_demos(agent_dir, refs)}")
