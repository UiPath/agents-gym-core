"""Create a standalone agent project without calling a model."""

import argparse
import json
import re
import shlex
import sys
import uuid
from pathlib import Path

from agents_gym_core.project import PROJECT_FILE

CALLABLE_LOOP = "agents_gym_core.loops.callable_loop.create_callable_graph"
MODEL = "gpt-4.1-mini-2025-04-14"

_ADAPTER = '''"""Replace run() with a call to your existing agent. Keep all four keyword arguments."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str


async def run(*, inputs: dict, system_prompt: str, user_prompt: str, model) -> dict:
    # Use these current prompts so ReAPO's edits reach your agent. Change Answer
    # and agent.yaml's schemas together when adapting a different task.
    response = await model.with_structured_output(Answer).ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt.format(**inputs)),
    ])
    return response.model_dump()
'''


def new_project(args: argparse.Namespace) -> None:
    name = args.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or "__" in name or name.endswith("_"):
        sys.exit(
            "gym new: use a name starting with a letter, then letters, digits, '-' or '_'; "
            "no '__' or trailing '_'"
        )
    root = Path(args.directory or name).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        sys.exit(f"gym new: {root} already exists; choose a new directory")
    agent = root / "agents" / name
    (agent / "prompts" / "baseline").mkdir(parents=True)
    (agent / "data").mkdir()
    (root / PROJECT_FILE).write_text(json.dumps({"version": 1, "agent": name}, indent=2) + "\n")
    (agent / "adapter.py").write_text(_ADAPTER)
    (root / ".env.example").write_text("# OPENAI_API_KEY=\nAGENTS_GYM_LLM_BACKEND=langchain\n")
    (root / ".gitignore").write_text(
        ".env\n__pycache__/\nexperiments/\ntmp_*/\n**/.gym_init_done\n"
    )
    (agent / "agent.yaml").write_text(f"""# @package _global_
version: base
loop: {CALLABLE_LOOP}
init:
  script: core::init.py
  data_dirs: [evaluations/eval-sets]
agent:
  entrypoint: core::adapter.py:run
  system_prompt: core::prompts/system_prompt.md
  user_prompt: core::prompts/user_prompt.md
  input_schema:
    fields:
      question: {{type: str}}
  output_schema:
    fields:
      answer: {{type: str}}
  evaluation_sets:
    - core::evaluations/eval-sets/evaluation-set-{name}_test.json
  tools: []
""")
    (agent / "versions.yaml").write_text(f"""train:
  version: train
  agent:
    evaluation_sets:
      - core::evaluations/eval-sets/evaluation-set-{name}_train.json
test:
  version: test
baseline-test:
  version: baseline-test
  agent:
    system_prompt: core::prompts/baseline/system_prompt.md
    user_prompt: core::prompts/baseline/user_prompt.md
""")
    (agent / "init.py").write_text(
        "from agents_gym_core.sdk.onboarding import prepare_jsonl as init\n"
    )
    for filename, prompt in (
        ("system_prompt.md", "Answer the question concisely. Return only the requested answer.\n"),
        ("user_prompt.md", "{question}\n"),
    ):
        (agent / "prompts" / filename).write_text(prompt)
        (agent / "prompts" / "baseline" / filename).write_text(prompt)
    for split, pairs in {
        "train": [("What is the capital of France?", "Paris"), ("What is 2 + 2?", "4")],
        "test": [("What is the capital of Italy?", "Rome"), ("What is 3 + 3?", "6")],
    }.items():
        rows = [
            {"id": f"{split}-{i}", "inputs": {"question": q}, "expected_output": {"answer": a}}
            for i, (q, a) in enumerate(pairs, 1)
        ]
        (agent / "data" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "README.md").write_text(f"""# {name}

Activate the framework's installed environment before running `gym`.
The included adapter and example cases demonstrate execution, scoring and optimization.
Connect your agent and add representative train/test data for your task.

1. Copy `.env.example` to `.env` and add model credentials.
2. Connect your agent in `agents/{name}/adapter.py`. Keep its keyword arguments
   and forward the current prompt arguments on every call.
3. Update the YAML schemas and `data/train.jsonl` / `data/test.jsonl` inside
   that agent directory. IDs and input objects must be disjoint across splits.
4. Commit your baseline (`git init` the project if needed), then run:

```bash
gym --project . init {name}
gym --project . eval --agents {name}:baseline-test --models {MODEL} --num_runs 1 --experiment_name baseline
gym --project . optimize {name} -m {MODEL} --optimizer-model gpt-5-2025-08-07 -r 2 -o
gym --project . eval --agents {name}:test --models {MODEL} --num_runs 1 --experiment_name optimized
```

Compare `experiments/{{baseline,optimized}}/final_score_per_agent.json`.
ReAPO edits current prompts using only `:train`; `:test` is held out.
Results, traces and round snapshots stay in this project. Use more repeats for
stochastic agents and a new experiment name after changing prompts or data.
After data edits, rebuild with `gym --project . init {name} --force`.
The template uses exact match; see the framework's evaluator guide for other scores.
""")
    print(
        f"Created {root}\n\n  cd {shlex.quote(str(root))}\n  gym --project . init {name}\n\nNext: follow README.md to connect your agent and evaluate its baseline."
    )


def prepare_jsonl(agent_dir: Path, params: dict) -> None:
    """Convert two small, explicit JSONL splits to Gym's evaluation format."""
    name = agent_dir.name
    evaluator = f"{name}_exact"
    payloads: dict[str, dict] = {}
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    # Validate both splits before writing either one.
    for split in ("train", "test"):
        path = agent_dir / "data" / f"{split}.jsonl"
        evaluations = []
        set_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gym:{name}:{split}"))
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"].strip()
            ):
                raise ValueError(f"{path}:{line_number}: each case needs a nonempty string id")
            if not isinstance(row.get("inputs"), dict) or not isinstance(
                row.get("expected_output"), dict
            ):
                raise ValueError(
                    f"{path}:{line_number}: inputs and expected_output must be objects"
                )
            identity = json.dumps(row["inputs"], sort_keys=True)
            # Run directories collide on case-insensitive filesystems.
            run_dir_id = str(row["id"]).replace("/", "_").lower()
            if run_dir_id in seen_ids or identity in seen_inputs:
                raise ValueError(
                    f"{path}:{line_number}: duplicate id or inputs across train/test cases"
                )
            seen_ids.add(run_dir_id)
            seen_inputs.add(identity)
            evaluations.append(
                {
                    "id": row["id"],
                    "name": row["id"],
                    "inputs": row["inputs"],
                    "evalSetId": set_id,
                    "evaluationCriterias": {evaluator: {"expectedOutput": row["expected_output"]}},
                }
            )
        if not evaluations:
            raise ValueError(f"{path}: split must contain at least one case")
        payloads[split] = {
            "version": "1.0",
            "id": set_id,
            "name": f"{name}_{split}",
            "evaluatorRefs": [evaluator],
            "evaluations": evaluations,
        }
    eval_root = agent_dir / "evaluations"
    (eval_root / "eval-sets").mkdir(parents=True, exist_ok=True)
    (eval_root / "evaluators").mkdir(exist_ok=True)
    for split, payload in payloads.items():
        (eval_root / "eval-sets" / f"evaluation-set-{name}_{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    evaluator_file = eval_root / "evaluators" / f"evaluator-{evaluator}.json"
    if not evaluator_file.exists():
        evaluator_file.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "id": evaluator,
                    "description": "Exact output match",
                    "evaluatorTypeId": "uipath-exact-match",
                    "evaluatorConfig": {
                        "name": "Exact match",
                        "targetOutputKey": "*",
                        "caseSensitive": False,
                    },
                },
                indent=2,
            )
            + "\n"
        )
    print(
        f"Prepared {name}: "
        + ", ".join(f"{s}={len(p['evaluations'])}" for s, p in payloads.items())
    )
