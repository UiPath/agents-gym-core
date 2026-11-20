"""gym optimize ends on the best round's prompts, whatever the optimizer left in place."""

import json
from pathlib import Path

import pytest

from agents_gym_core.prompt_optimization import best_round
from agents_gym_core.prompt_optimization.mcp_tools import helpers, tools_runner
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider


@pytest.fixture
def agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    agent = tmp_path / "agents" / "toy"
    (agent / "prompts").mkdir(parents=True)
    (agent / "agent.yaml").write_text(
        "agent:\n"
        "  system_prompt: core::prompts/system_prompt.md\n"
        "  user_prompt: core::prompts/user_prompt.md\n"
    )
    (agent / "prompts" / "user_prompt.md").write_text("{query}")
    provider = CoreDirectoryProvider()
    monkeypatch.setattr(provider, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(provider, "get_agents_dir", lambda: tmp_path / "agents")
    monkeypatch.setattr(helpers, "DIRECTORY_PROVIDER", provider)
    return agent


def _evaluate_round(workspace: Path, n: int, score: float, agent_dir: Path, system: str) -> None:
    (agent_dir / "prompts" / "system_prompt.md").write_text(system)
    round_dir = workspace / f"round_{n:03d}"
    best_round.snapshot_prompts(round_dir, "toy")
    (round_dir / "round_stats.json").write_text(json.dumps({"score": score}))


def test_the_best_round_is_restored_not_the_last(tmp_path: Path, agent_dir: Path) -> None:
    workspace = tmp_path / "workspace"
    _evaluate_round(workspace, 0, 0.70, agent_dir, "baseline")
    _evaluate_round(workspace, 1, 0.80, agent_dir, "round 1")
    _evaluate_round(workspace, 2, 0.75, agent_dir, "round 2")  # the optimizer stops here

    result = best_round.restore_best_round(workspace, "toy")

    assert result == {"round": "round_001", "score": 0.80, "restored": ["prompts/system_prompt.md"]}
    assert (agent_dir / "prompts" / "system_prompt.md").read_text() == "round 1"
    assert json.loads((workspace / "best_round.json").read_text()) == result


def test_ties_go_to_the_earliest_round_including_the_baseline(
    tmp_path: Path, agent_dir: Path
) -> None:
    workspace = tmp_path / "workspace"
    _evaluate_round(workspace, 0, 0.80, agent_dir, "baseline")
    _evaluate_round(workspace, 1, 0.80, agent_dir, "round 1")

    result = best_round.restore_best_round(workspace, "toy")

    assert result is not None and result["round"] == "round_000"
    assert (agent_dir / "prompts" / "system_prompt.md").read_text() == "baseline"


def test_only_configured_prompt_files_are_restored(tmp_path: Path, agent_dir: Path) -> None:
    workspace = tmp_path / "workspace"
    _evaluate_round(workspace, 0, 0.90, agent_dir, "baseline")
    _evaluate_round(workspace, 1, 0.50, agent_dir, "round 1")
    planted = workspace / "round_000" / best_round.SNAPSHOT_DIR
    (planted / "agent.yaml").write_text("agent: {}\n")
    (planted / "tools.py").write_text("import os\n")

    result = best_round.restore_best_round(workspace, "toy")

    assert result is not None and result["restored"] == ["prompts/system_prompt.md"]
    assert "system_prompt" in (agent_dir / "agent.yaml").read_text()
    assert not (agent_dir / "tools.py").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "round_001/evaluated_prompts/prompts/system_prompt.md",
        "round_001/Evaluated_Prompts/x.md",
        "round_001/evaluated_prompts. /x.md",
    ],
)
def test_the_optimizer_cannot_write_a_snapshot(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_best_round_check")

    assert tools_runner.save_optimizer_log("", filename, "forged").startswith("Refused")
