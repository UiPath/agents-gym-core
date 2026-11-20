"""GEPA runners end to end, offline: one candidate mutation per variant against a local stub.

Runs native GEPA and GEPA-DSPy on ``claim_history:train`` through the public backend with no
UiPath credentials (tests/fixtures/openai_stub_server.py stands in for the provider). The
stub proposes a fixed instruction on every reflection call. Asserts that:
- the run finishes and the proposed instruction reaches later agent rollouts;
- no datapoint of the held-out test split is ever sent to a model;
- the agent's prompt files are untouched (``--no-writeback``).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "src" / "agents_gym_core" / "agents" / "claim_history"
PROPOSED = "Solve the arithmetic expression and return the integer result."
AGENT_MODEL = "gpt-4.1-mini-2025-04-14"

VARIANTS = {
    "native": [
        "agents_gym_core.prompt_optimization.gepa",
        "--proposer-model",
        "gpt-5-2025-08-07",
        "--num-iters",
        "1",
        "--minibatch",
        "2",
    ],
    "dspy": [
        "agents_gym_core.prompt_optimization.gepa_dspy",
        "--reflection-model",
        "gpt-5-2025-08-07",
        "--reflection-temperature",
        "1.0",
        "--max-metric-calls",
        "12",
        "--minibatch",
        "2",
        "--no-merge",
        "--num-threads",
        "2",
    ],
}


def _prompt_hashes() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((AGENT_DIR / "prompts").glob("*.md"))
    }


def _test_split_signatures() -> list[str]:
    split = AGENT_DIR / "evaluations" / "eval-sets" / "evaluation-set-claim_history_test.json"
    return [
        f"Denial Code: {e['inputs']['DenialCode']}\nRemittance Code: {e['inputs']['RemittanceCode']}"
        for e in json.loads(split.read_text())["evaluations"]
    ]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_gepa_mutation_reaches_rollouts_and_never_sees_test_split(
    variant: str, stub: tuple[int, Path], tmp_path: Path
) -> None:
    port, log = stub
    prompts_before = _prompt_hashes()
    module, *options = VARIANTS[variant]
    run = subprocess.run(
        [sys.executable, "-m", module, "claim_history", "--version", "train"]
        + ["--model", AGENT_MODEL, "--no-writeback", "--run-dir", str(tmp_path / "runs")]
        + options,
        cwd=tmp_path,  # the UiPath SDK loads <cwd>/.env with override=True at import
        env={  # no UIPATH_* / LLMGW_* variables
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "AGENTS_GYM_LLM_BACKEND": "langchain",
            "AGENTS_GYM_SKIP_DOTENV": "1",
            "OPENAI_API_KEY": "sk-stub",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        },
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    requests = [json.loads(line) for line in log.read_text().splitlines()]
    rollouts = [r for r in requests if r["model"] == AGENT_MODEL]
    assert rollouts, "no agent rollouts reached the stub"
    assert any(PROPOSED in r["text"] for r in rollouts), "proposed instruction never rolled out"
    for signature in _test_split_signatures():
        leaked = [r["model"] for r in requests if signature in r["text"]]
        assert not leaked, f"test datapoint sent to {leaked}: {signature!r}"
    assert _prompt_hashes() == prompts_before
