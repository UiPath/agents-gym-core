"""Every `gym` subcommand and mode, end to end and offline (`gym eval`: test_public_backend_e2e).

The ``stub`` fixture (an OpenAI-compatible server) answers every model call. The UiPath
commands run from a scratch project whose ``gym_generate.json`` points at this repo's agents.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / "src" / "agents_gym_core" / "agents"
MODEL = "gpt-4.1-mini-2025-04-14"


def _env(home: Path, port: int | None = None) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "AGENTS_GYM_SKIP_DOTENV": "1"}
    if port is not None:
        env |= {
            "AGENTS_GYM_LLM_BACKEND": "langchain",
            "OPENAI_API_KEY": "sk-stub",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        }
    return env


def _gym(
    *args: str, cwd: Path, env: dict[str, str], ok: bool = True
) -> subprocess.CompletedProcess:
    run = subprocess.run(
        [sys.executable, "-m", "agents_gym_core.sdk.cli", *args],
        cwd=cwd,  # the UiPath SDK loads <cwd>/.env with override=True
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if ok:
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    return run


def _requests(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("command", ["eval", "optimize", "reproduce"])
@pytest.mark.parametrize("flag", ["--help", "--unknown-option"])
def test_forwarded_usage_names_the_public_command(tmp_path: Path, command: str, flag: str) -> None:
    help_requested = flag == "--help"
    result = _gym(command, flag, cwd=tmp_path, env=_env(tmp_path), ok=help_requested)
    assert result.returncode == (0 if help_requested else 2)
    assert f"usage: gym {command} " in (result.stdout if help_requested else result.stderr)


@pytest.mark.parametrize("argument", ["--help", "--unknown-option"])
def test_root_help_and_errors_do_not_load_the_agent_stack(tmp_path: Path, argument: str) -> None:
    code = """
import sys
from agents_gym_core.sdk.cli import main
try:
    main([sys.argv[1]])
except SystemExit:
    pass
assert "agents_gym_core.project" not in sys.modules
assert "agents_gym_core.utils.directory_provider" not in sys.modules
assert "omegaconf" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, argument],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_init_lists_and_runs_init_scripts(tmp_path: Path) -> None:
    env = _env(tmp_path)
    listing = _gym("init", "--list", cwd=tmp_path, env=env).stdout
    for agent in ("hotpotqa", "hover", "aime_2025_agent", "livebench_math", "ifbench", "papillon"):
        assert agent in listing
    assert "nothing to do" in _gym("init", "calculator", cwd=tmp_path, env=env).stdout
    assert _gym("init", cwd=tmp_path, env=env, ok=False).returncode != 0


@pytest.mark.parametrize("fails", [False, True])
def test_uipath_init_preserves_existing_diagrams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    from agents_gym_core.sdk import cli

    existing = tmp_path / "architecture.mermaid"
    removed = tmp_path / "notes.mermaid"
    generated = tmp_path / "calculator.mermaid"
    existing.write_bytes(b"graph TD\n user --> service\n")
    removed.write_bytes(b"graph LR\n input --> output\n")

    def initialize(*args: object, **kwargs: object) -> None:
        existing.write_text("generated replacement")
        removed.unlink()
        generated.write_text("generated diagram")
        if fails:
            raise subprocess.CalledProcessError(1, ["uipath", "init"])

    monkeypatch.setattr(cli, "_uipath", lambda *args: ["uipath", *args])
    monkeypatch.setattr(cli.subprocess, "run", initialize)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            cli._uipath_init(tmp_path)
    else:
        cli._uipath_init(tmp_path)

    assert existing.read_bytes() == b"graph TD\n user --> service\n"
    assert removed.read_bytes() == b"graph LR\n input --> output\n"
    assert not generated.exists()


def test_uipath_generate_run_and_eval(stub: tuple[int, Path], tmp_path: Path) -> None:
    port, log = stub
    project = tmp_path / "project"
    project.mkdir()
    registry = "from agents_gym_core.hydra_agent import CoreAgentRegistry"
    (project / "gym_generate.json").write_text(
        json.dumps({"agents_dirs": [str(AGENTS)], "registry_import": registry})
    )
    (project / "pyproject.toml").write_text(
        '[project]\nname = "gym-cli-test"\nversion = "0.0.0"\ndescription = "test"\n'
    )
    (project / "CLAUDE.md").write_text("my notes\n")
    env = _env(tmp_path, port)
    entry_points = project / "entry-points.json"

    # run and eval write the entry points first when they are missing
    _gym("uipath", "run", "calculator", cwd=project, env=env)
    graphs = json.loads((project / "langgraph.json").read_text())["graphs"]
    assert graphs == {"calculator": "./sdk_entrypoints.py:calculator_agent"}
    assert "calculator" in entry_points.read_text()
    assert (project / "CLAUDE.md").read_text() == "my notes\n"
    assert not (project / "AGENTS.md").exists()

    _gym("uipath", "generate", "--agents", "calculator", "--with-evaluations", cwd=project, env=env)
    assert (project / "evaluations" / "eval-sets" / "evaluation-set-calculator.json").exists()
    assert (project / "CLAUDE.md").read_text() == "my notes\n"
    written = entry_points.stat().st_mtime_ns

    by_index = _gym("uipath", "run", "calculator", "1", cwd=project, env=env).stdout
    assert "Running datapoint 1 of evaluation-set-calculator.json" in by_index
    question = '{"expression": "how much is 40 + 2"}'
    _gym("uipath", "run", "calculator", question, cwd=project, env=env)
    assert any("how much is 40 + 2" in r["text"] for r in _requests(log))

    # the default eval set is the first one agent.yaml lists
    for flags, name in (
        ([], "calculator"),
        (["--eval-set", "calculator-small"], "calculator-small"),
    ):
        _gym("uipath", "eval", "calculator", *flags, "--workers", "4", cwd=project, env=env)
        results = project / "experiments" / "sdk_results" / f"eval-results-{name}.json"
        assert json.loads(results.read_text())["evaluationSetResults"]
    assert entry_points.stat().st_mtime_ns == written  # up to date, so not regenerated

    missing = _gym("uipath", "run", "calculator", "99", cwd=project, env=env, ok=False)
    assert missing.returncode != 0 and "out of range" in missing.stderr


@pytest.mark.skipif(
    (AGENTS / "hover" / ".gym_init_done").exists(),
    reason="hover is initialised here; the check needs an agent without its gym init data",
)
def test_uipath_commands_ask_for_gym_init_first(tmp_path: Path) -> None:
    for args in (("run", "hover"), ("eval", "hover"), ("generate", "--agents", "hover")):
        refused = _gym("uipath", *args, cwd=tmp_path, env=_env(tmp_path), ok=False)
        assert refused.returncode != 0 and "gym init" in refused.stderr, refused.stderr


def test_optimize_runs_the_react_engine_through_the_tool_server(
    stub: tuple[int, Path], tmp_path: Path
) -> None:
    # The stub keeps calling the first tool it is offered, so the run ends at the step cap.
    port, log = stub
    name = "pytest_gym_cli"
    try:
        run = _gym(
            "optimize", "calculator", "--optimizer-model", MODEL, "-m", MODEL,
            "-r", "1", "--recursion-limit", "12", "-n", name, "-o",
            cwd=tmp_path, env=_env(tmp_path, port), ok=False,
        )  # fmt: skip
        assert run.returncode == 1 and "Stopped at --recursion-limit 12" in run.stderr, run.stderr
        (workspace,) = REPO.glob(f"tmp_{name}_calculator_*")
        metadata = json.loads((workspace / "run_metadata.json").read_text())
        assert (metadata["status"], metadata["error_type"]) == ("failed", "GraphRecursionError")
        optimizer_calls = [r for r in _requests(log) if r["model"] == MODEL]
        cost = metadata["optimizer_cost"]  # the stub reports 1 input and 1 output token per call
        assert cost["num_llm_calls"] == cost["input_tokens"] == len(optimizer_calls) > 0
        assert (
            cost["by_model"][MODEL]["cost_usd"] > 0 and metadata["eval_cost"]["num_llm_calls"] == 0
        )
        assert metadata["total_cost_usd"] == cost["total_cost_usd"]
        assert metadata["total_duration_s"] > 0 and "Optimizer:" in run.stdout
        offered = {tuple(r["tools"]) for r in optimizer_calls}
        assert len(offered) == 1 and len(next(iter(offered))) == 20
        assert {"run_agent_async", "update_prompt", "get_full_results"} <= set(*offered)
    finally:
        for workspace in REPO.glob(f"tmp_{name}_*"):
            shutil.rmtree(workspace)


CLAUDE_RESULT = {  # the end of a Claude CLI stream-json run, as it reports usage and list prices
    "type": "result",
    "is_error": False,
    "num_turns": 17,
    "total_cost_usd": 0.26,
    "modelUsage": {
        "claude-opus-5-5": {
            "inputTokens": 16,
            "outputTokens": 3444,
            "cacheReadInputTokens": 112651,
            "cacheCreationInputTokens": 19783,
            "costUSD": 0.25,
        },
        "claude-sonnet-4-6": {"inputTokens": 3152, "outputTokens": 11, "costUSD": 0.01},
    },
}


@pytest.mark.parametrize("fails", [False, True], ids=["completes", "login expired"])
def test_optimize_claude_engine_records_the_cli_cost(tmp_path: Path, fails: bool) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    result = CLAUDE_RESULT | (
        {"is_error": True, "result": "Failed to authenticate", "num_turns": 1, "total_cost_usd": 0}
        | {"modelUsage": {}}
        if fails
        else {}
    )
    fake = bin_dir / "claude"
    fake.write_text(f"#!/bin/sh\necho '{json.dumps(result)}'\n")
    fake.chmod(0o755)
    env = _env(tmp_path) | {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    name = "pytest_claude_cost"
    try:
        run = _gym(
            "optimize", "calculator", "--engine", "claude", "-r", "1", "-n", name, "-o",
            cwd=tmp_path, env=env, ok=False,
        )  # fmt: skip
        (workspace,) = REPO.glob(f"tmp_{name}_*")
        metadata = json.loads((workspace / "run_metadata.json").read_text())
        cost = metadata["optimizer_cost"]
        if fails:
            assert run.returncode == 1, run.stdout[-2000:] + run.stderr[-2000:]
            assert "the claude CLI failed: Failed to authenticate" in run.stderr
            assert (metadata["status"], cost["total_cost_usd"]) == ("failed", 0.0)
            return
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
        assert metadata["status"] == "completed" and "Optimizer:" in run.stdout
        assert (cost["total_cost_usd"], cost["num_llm_calls"]) == (0.26, 17)
        opus = cost["by_model"]["claude-opus-5-5"]
        assert (opus["input_tokens"], opus["cached_tokens"]) == (16 + 112651 + 19783, 112651)
        assert metadata["total_cost_usd"] == 0.26 and metadata["eval_cost"]["num_llm_calls"] == 0
    finally:
        for workspace in REPO.glob(f"tmp_{name}_*"):
            shutil.rmtree(workspace)
