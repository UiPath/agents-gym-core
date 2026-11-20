"""Writes the train/test splits of the full eval set (default 60/40, seed 42)."""

import copy
import json
import random
import uuid
from pathlib import Path

AGENT = "claim_history"


def init(agent_dir: Path, params: dict) -> None:
    seed = params.get("seed", 42)
    train_ratio = params.get("train_ratio", 0.6)

    eval_dir = agent_dir / "evaluations" / "eval-sets"
    source = eval_dir / f"evaluation-set-{AGENT}.json"

    with open(source) as f:
        data = json.load(f)

    evals = data["evaluations"]
    rng = random.Random(seed)
    indices = list(range(len(evals)))
    rng.shuffle(indices)

    split = int(len(evals) * train_ratio)
    train_indices = sorted(indices[:split])
    test_indices = sorted(indices[split:])

    for label, idx_list in [("train", train_indices), ("test", test_indices)]:
        split_data = copy.deepcopy(data)
        split_data["evaluations"] = [evals[i] for i in idx_list]
        split_data["name"] = f"evaluation-set-{AGENT}_{label}"
        split_data["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{AGENT}_{label}_seed{seed}"))

        path = eval_dir / f"evaluation-set-{AGENT}_{label}.json"
        path.write_text(json.dumps(split_data, indent=2) + "\n")
        print(f"Wrote {path.name}  ({len(idx_list)} datapoints)")
