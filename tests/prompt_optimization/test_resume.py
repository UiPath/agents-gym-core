import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.main import (
    detect_last_completed_react_round,
    resolve_resume,
)
from agents_gym_core.prompt_optimization.mcp_tools import helpers, tools_runner
from agents_gym_core.prompt_optimization.react_optimizer import optimizer
from agents_gym_core.prompt_optimization.react_optimizer.optimizer import (
    CHECKPOINT_FILENAME,
    THREAD_STATE_FILENAME,
    _resolve_thread_id,
)


@pytest.mark.parametrize("completed_round", [False, True])
def test_resume_prompt_uses_selected_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    completed_round: bool,
) -> None:
    (tmp_path / "gym-project.json").write_text('{"version": 1}')
    workspace = tmp_path / "tmp_existing"
    workspace.mkdir()
    if completed_round:
        (workspace / "round_000").mkdir()
        (workspace / "round_000" / "round_stats.json").write_text('{"score": 0.8}')
    monkeypatch.setattr("builtins.input", lambda prompt: "c")

    with helpers.optimization_context(ProjectDirectoryProvider(tmp_path)):
        assert resolve_resume(workspace, "") == (int(completed_round), "continue")

    output = capsys.readouterr().out
    assert "Existing workspace found: tmp_existing/" in output
    assert ("Rounds 0..0 already exist." in output) is completed_round
    assert "0..-1" not in output


def test_react_thread_is_reused_only_when_checkpoint_exists(tmp_path: Path) -> None:
    first_thread, first_mode = _resolve_thread_id(tmp_path, start_round=0)

    assert first_mode == "new"
    assert json.loads((tmp_path / THREAD_STATE_FILENAME).read_text())["thread_id"] == first_thread

    # A pre-checkpoint workspace can only reconstruct context from round files.
    bootstrap_thread, bootstrap_mode = _resolve_thread_id(tmp_path, start_round=3)
    assert bootstrap_mode == "artifact_bootstrap"
    assert bootstrap_thread != first_thread

    (tmp_path / CHECKPOINT_FILENAME).touch()
    resumed_thread, resumed_mode = _resolve_thread_id(tmp_path, start_round=4)

    assert resumed_mode == "checkpoint"
    assert resumed_thread == bootstrap_thread


def test_fresh_react_run_rotates_thread_id(tmp_path: Path) -> None:
    old_thread, _ = _resolve_thread_id(tmp_path, start_round=0)
    (tmp_path / CHECKPOINT_FILENAME).touch()

    new_thread, mode = _resolve_thread_id(tmp_path, start_round=0)

    assert mode == "new"
    assert new_thread != old_thread


def test_react_resume_reuses_incomplete_round_number(tmp_path: Path) -> None:
    (tmp_path / "round_000").mkdir()
    (tmp_path / "round_000" / "round_stats.json").write_text('{"score": 0.8}')
    (tmp_path / "round_001").mkdir()
    (tmp_path / "round_001" / "summarize1.md").write_text("candidate")

    assert detect_last_completed_react_round(tmp_path) == 0


@pytest.mark.parametrize("prior_full_round", [False, True])
def test_react_resume_keeps_a_round_with_only_partial_evaluations_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_full_round: bool
) -> None:
    monkeypatch.setenv("COACH_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tools_runner, "_eval_run_record", lambda info: {"duration_s": 1})
    if prior_full_round:
        (tmp_path / "round_000").mkdir()
        (tmp_path / "round_000" / "round_stats.json").write_text('{"score": 0.0}')
    partial_round = int(prior_full_round)
    tools_runner._write_round_stats(
        {
            "experiment_name": f"review_round_{partial_round:03d}_001",
            "max_datapoints": 1,
        },
        "toy",
        1,
        1.0,
    )

    assert detect_last_completed_react_round(tmp_path) + 1 == partial_round


@pytest.mark.parametrize(
    "stats", ["{}", '{"score": null}', '{"score": NaN}', '{"score": true}', "{"]
)
def test_react_resume_ignores_unscored_or_invalid_statistics(tmp_path: Path, stats: str) -> None:
    (tmp_path / "round_000").mkdir()
    (tmp_path / "round_000" / "round_stats.json").write_text(stats)

    assert detect_last_completed_react_round(tmp_path) == -1


def test_continue_before_round0_stats_keeps_checkpointed_thread(tmp_path: Path) -> None:
    """An interruption during round 0 leaves a checkpoint but no round_stats.json."""
    thread_id, _ = _resolve_thread_id(tmp_path, start_round=0)
    (tmp_path / CHECKPOINT_FILENAME).touch()
    (tmp_path / "round_000").mkdir()
    state_before = (tmp_path / THREAD_STATE_FILENAME).read_text()

    start_round = detect_last_completed_react_round(tmp_path) + 1
    resumed, mode = _resolve_thread_id(tmp_path, start_round=start_round, resume=True)

    assert start_round == 0
    assert (resumed, mode) == (thread_id, "checkpoint")
    assert (tmp_path / THREAD_STATE_FILENAME).read_text() == state_before


async def test_resume_continues_pending_node_without_replaying_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--continue after a round-0 interruption must not re-send the protocol prompt."""
    output_dir = "tmp_resume_pending"
    workspace = tmp_path / output_dir
    workspace.mkdir()
    old_thread, _ = _resolve_thread_id(workspace, start_round=0)
    (workspace / CHECKPOINT_FILENAME).touch()

    calls: dict[str, Any] = {}

    class FakeAgent:
        async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
            calls["state_thread"] = config["configurable"]["thread_id"]
            return SimpleNamespace(next=("tools",))  # stopped before a pending tool node

        async def astream(self, stream_input: object, config: dict[str, Any], stream_mode: str):
            calls["stream_input"] = stream_input
            calls["stream_thread"] = config["configurable"]["thread_id"]
            return
            yield  # pragma: no cover - makes this an async generator

    @asynccontextmanager
    async def no_tools():
        yield []

    monkeypatch.setattr(optimizer, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(helpers, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(optimizer, "get_chat_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(optimizer, "coach_light_session", no_tools)
    monkeypatch.setattr(optimizer, "create_react_agent", lambda **kwargs: FakeAgent())

    await optimizer.run_react_optimizer(
        "claim_history:train",
        output_dir=output_dir,
        num_rounds=1,
        skill_version="full",
        resume=True,
    )

    assert calls["state_thread"] == calls["stream_thread"] == old_thread
    assert calls["stream_input"] is None
    metadata = json.loads((workspace / "run_metadata.json").read_text())
    assert (metadata["resume_mode"], metadata["start_round"]) == ("checkpoint", 0)
