"""Generate evaluation set JSONs for aime_2025_agent.

Train: AIME 2022-2024 from AI-MO/aimo-validation-aime (90 problems, with solutions)
Test:  AIME 2025 from MathArena/aime_2025 (30 problems)

Modes:
  --train      AIME 2022-2024  ->  evaluation-set-aime_2022_2024.json
  --test       AIME 2025       ->  evaluation-set-aime_2025.json
  --split      AIME 2022-2024 shuffled (seed=0) and split 50/50:
                 ->  evaluation-set-aime_train.json  (45 problems)
                 ->  evaluation-set-aime_val.json    (45 problems)
               GEPA's train/val split (gepa_artifact/benchmarks/AIME/AIME_data.py).

Usage:
    python src/agents_gym_core/agents/aime_2025_agent/scripts/build_dataset.py --train
    python src/agents_gym_core/agents/aime_2025_agent/scripts/build_dataset.py --test
    python src/agents_gym_core/agents/aime_2025_agent/scripts/build_dataset.py --split
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

# Pinned so the splits stay byte-identical.
AIME_2025_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"
AIMO_AIME_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"

EVAL_SET_ID = "aime-eval-001"
EVALUATOR_REF = "aime_2025_ExactMatchEvaluator"
EVAL_SETS_DIR = Path(__file__).resolve().parent.parent / "evaluations" / "eval-sets"


def extract_answer(raw: str) -> int:
    """Extract the leading integer of an answer string ('075' -> 75)."""
    match = re.match(r"(\d+)", str(raw).strip())
    if match:
        return int(match.group(1))
    return int(raw)


def build(
    data: list[dict],
    output_path: Path,
    eval_set_name: str,
    id_prefix: str,
    question_key: str,
    answer_key: str,
    solution_key: str | None = None,
    repeats: int = 1,
) -> None:
    # Round-robin ordering to match GEPA's `test_split * 5`:
    # [p0, p1, ..., pN, p0, p1, ..., pN, ...] instead of grouped [p0x5, p1x5, ...]
    evaluations = [
        {
            "id": f"{id_prefix}-{idx + 1:03d}" + (f"-r{r}" if repeats > 1 else ""),
            "name": f"{id_prefix}-{idx + 1:03d}" + (f"-r{r}" if repeats > 1 else ""),
            "inputs": {
                "problem": item[question_key],
            },
            "expectedOutput": {
                **(
                    {"justification": item[solution_key]}
                    if solution_key and item.get(solution_key)
                    else {}
                ),
            },
            "simulationInstructions": "",
            "expectedAgentBehavior": item.get(solution_key, "") if solution_key else "",
            "simulateInput": False,
            "inputGenerationInstructions": "",
            "simulateTools": False,
            "toolsToSimulate": [],
            "evalSetId": EVAL_SET_ID,
            "createdAt": "2026-03-27T00:00:00.000Z",
            "updatedAt": "2026-03-27T00:00:00.000Z",
            "source": "aime_dataset",
            "evaluationCriterias": {
                EVALUATOR_REF: {
                    "expectedOutput": {
                        "answer": extract_answer(item[answer_key]),
                        **(
                            {"reasoning": item[solution_key]}
                            if solution_key and item.get(solution_key)
                            else {}
                        ),
                    },
                    "expectedAgentBehavior": item.get(solution_key, "") if solution_key else "",
                }
            },
        }
        for r in range(1, repeats + 1)
        for idx, item in enumerate(data)
    ]

    eval_set = {
        "version": "1.0",
        "id": EVAL_SET_ID,
        "name": eval_set_name,
        "evaluatorRefs": [EVALUATOR_REF],
        "modelSettings": [],
        "createdAt": "2026-03-27T00:00:00.000Z",
        "updatedAt": "2026-03-27T00:00:00.000Z",
        "evaluations": evaluations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Saved {len(evaluations)} evaluations -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="AIME 2022-2024 (90 problems).")
    group.add_argument("--test", action="store_true", help="AIME 2025 (30 problems).")
    group.add_argument(
        "--split",
        action="store_true",
        help="AIME 2022-2024 shuffled (seed=0), split 50/50 into train (45) and val (45). "
        "GEPA's split.",
    )
    args = parser.parse_args()

    if args.split:
        print("Loading AIME 2022-2024 from AI-MO/aimo-validation-aime...")
        data = load_dataset(
            "AI-MO/aimo-validation-aime", split="train", revision=AIMO_AIME_REVISION
        ).to_list()
        print(f"  {len(data)} problems")
        # Shuffle with seed=0, as GEPA's AIME_data.py does
        random.Random(0).shuffle(data)
        mid = len(data) // 2
        train_data, val_data = data[:mid], data[mid:]
        print(f"  Split: {len(train_data)} train, {len(val_data)} val")
        build(
            train_data,
            EVAL_SETS_DIR / "evaluation-set-aime_train.json",
            "evaluation-set-aime_train",
            "aime-train",
            question_key="problem",
            answer_key="answer",
            solution_key="solution",
        )
        build(
            val_data,
            EVAL_SETS_DIR / "evaluation-set-aime_val.json",
            "evaluation-set-aime_val",
            "aime-val",
            question_key="problem",
            answer_key="answer",
            solution_key="solution",
        )
    elif args.train:
        print("Loading AIME 2022-2024 from AI-MO/aimo-validation-aime...")
        data = load_dataset(
            "AI-MO/aimo-validation-aime", split="train", revision=AIMO_AIME_REVISION
        ).to_list()
        print(f"  {len(data)} problems")
        build(
            data,
            EVAL_SETS_DIR / "evaluation-set-aime_2022_2024.json",
            "evaluation-set-aime_2022_2024",
            "aime-2022-2024",
            question_key="problem",
            answer_key="answer",
            solution_key="solution",
        )
    else:
        print("Loading AIME 2025 from MathArena/aime_2025...")
        data = load_dataset(
            "MathArena/aime_2025", split="train", revision=AIME_2025_REVISION
        ).to_list()
        print(f"  {len(data)} problems")
        build(
            data,
            EVAL_SETS_DIR / "evaluation-set-aime_2025.json",
            "evaluation-set-aime_2025",
            "aime-2025",
            question_key="problem",
            answer_key="answer",
        )


if __name__ == "__main__":
    main()
