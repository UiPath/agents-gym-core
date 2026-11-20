"""A variable set in the shell wins over the same key in ``.env``."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_shell_value_wins_and_new_dotenv_keys_load(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "GYM_TEST_SHELL_KEY=from_dotenv\nGYM_TEST_DOTENV_ONLY=from_dotenv\n"
    )
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", str(tmp_path))}
    env["GYM_TEST_SHELL_KEY"] = "from_shell"
    code = (
        "import os, agents_gym_core; "
        "print(os.environ['GYM_TEST_SHELL_KEY'], os.environ.get('GYM_TEST_DOTENV_ONLY'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["from_shell", "from_dotenv"]


@pytest.mark.parametrize("mode", ["project", "shell", "skip"])
def test_project_dotenv_precedence_reaches_mcp_workers(tmp_path: Path, mode: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "gym-project.json").write_text('{"version": 1}')
    (project / ".env").write_text(
        "OPENAI_BASE_URL=http://project.invalid/v1\nGYM_TEST_PROJECT_ONLY=project\n"
    )
    (tmp_path / ".env").write_text("OPENAI_BASE_URL=http://cwd.invalid/v1\n")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LITELLM_MODE": "PRODUCTION"}
    if mode == "shell":
        env["OPENAI_BASE_URL"] = "http://shell.invalid/v1"
    elif mode == "skip":
        env["AGENTS_GYM_SKIP_DOTENV"] = "1"
    code = """
import json, os, sys
from agents_gym_core.sdk import cli
def observe(args):
    from agents_gym_core.prompt_optimization.mcp_tools.helpers import optimization_context
    from agents_gym_core.prompt_optimization.mcp_tools.mcp_client import _build_connection
    keys = ("OPENAI_BASE_URL", "GYM_TEST_PROJECT_ONLY")
    with optimization_context(args.directory_provider):
        worker_env = _build_connection().env
    print(json.dumps([[os.environ.get(k) for k in keys], [worker_env.get(k) for k in keys]]))
cli._forward = observe
cli.main(["--project", sys.argv[1], "eval"])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(project)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    expected = [None, None] if mode == "skip" else [f"http://{mode}.invalid/v1", "project"]
    assert json.loads(result.stdout) == [expected, expected]
