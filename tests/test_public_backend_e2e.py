"""End to end in public mode, offline: a testbed run with no UiPath or gateway credentials.

A local OpenAI-compatible stub (tests/fixtures/openai_stub_server.py) stands in for the
provider. The run must finish, the agent and its LLM judge must both reach the stub, and the
judge must keep its required tool choice and return a score rather than an exception.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT_MODEL = "gpt-4.1-mini-2025-04-14"
JUDGE_MODEL = "gpt-4o-2024-11-20"  # calculator's LLMJudgeOutputEvaluator


def test_calculator_runs_without_gateway_credentials(
    stub: tuple[int, Path], tmp_path: Path
) -> None:
    port, log = stub
    env = {  # deliberately no UIPATH_* / LLMGW_* variables
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "AGENTS_GYM_LLM_BACKEND": "langchain",
        "AGENTS_GYM_SKIP_DOTENV": "1",
        "OPENAI_API_KEY": "sk-stub",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
    }
    run = subprocess.run(
        [sys.executable, "-m", "agents_gym_core.sdk.cli", "eval", "--agents", "calculator"]
        + ["--models", AGENT_MODEL, "--max_datapoints", "1"]
        + ["--experiment_name", "public_e2e", "--output_dir", str(tmp_path / "experiments")],
        cwd=tmp_path,  # the UiPath SDK loads <cwd>/.env with override=True at import
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]

    requests = [json.loads(line) for line in log.read_text().splitlines()]
    agent = [r for r in requests if r["model"] == AGENT_MODEL]
    judge = [r for r in requests if r["model"] == JUDGE_MODEL]
    assert agent and all(r["kind"] == "tool_call:end_execution" for r in agent)
    assert judge and all(r["tool_choice"] == "required" for r in judge)

    experiment = tmp_path / "experiments" / "public_e2e"
    results = list(experiment.rglob("calculator_LLMJudgeOutputEvaluator.json"))
    assert results
    for path in results:
        result = json.loads(path.read_text())
        assert "Exception" not in str(result["details"]), result["details"]
        assert result["score"] == pytest.approx(0.01)  # the stub's judge scores 1 out of 100

    scores = json.loads((experiment / "final_score_per_agent.json").read_text())
    assert [set(by_agent) for by_agent in scores.values()] == [{"calculator__base"}]
