from pathlib import Path

import pytest

from agents_gym_core.prompt_optimization.mcp_tools import tools_runner


def test_continuation_cannot_overwrite_prior_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_runner, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_resume_test")
    monkeypatch.setenv("COACH_IMMUTABLE_ROUNDS_BEFORE", "3")

    prior_file = tmp_path / "tmp_resume_test" / "round_002" / "scores.md"
    prior_file.parent.mkdir(parents=True)
    prior_file.write_text("original")

    result = tools_runner.save_optimizer_log("", "round_002/scores.md", "replacement")

    assert result.startswith("Refused: round_002 is immutable")
    assert prior_file.read_text() == "original"


def test_continuation_can_write_current_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_runner, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_resume_test")
    monkeypatch.setenv("COACH_IMMUTABLE_ROUNDS_BEFORE", "3")

    result = tools_runner.save_optimizer_log("", "round_003/scores.md", "new score")

    assert result.startswith("Saved ")
    assert (tmp_path / "tmp_resume_test" / "round_003" / "scores.md").read_text() == "new score"


@pytest.mark.parametrize(
    "alias",
    ["./round_000/scores.md", "round_000/./scores.md", "round_001/../round_000/scores.md"],
)
def test_immutable_round_cannot_be_aliased(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    monkeypatch.setattr(tools_runner, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_resume_test")
    monkeypatch.setenv("COACH_IMMUTABLE_ROUNDS_BEFORE", "1")
    prior_file = tmp_path / "tmp_resume_test" / "round_000" / "scores.md"
    prior_file.parent.mkdir(parents=True)
    prior_file.write_text("original")

    result = tools_runner.save_optimizer_log("", alias, "replacement")

    assert result.startswith("Refused:"), result
    assert prior_file.read_text() == "original"
    assert tools_runner.save_optimizer_log("", "./round_001/scores.md", "ok").startswith("Saved ")
