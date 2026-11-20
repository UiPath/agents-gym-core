"""The Claude engine must launch with this checkout's coach-light server, not a machine-global one."""

import argparse
import json
import sys
from pathlib import Path

import pytest

from agents_gym_core.prompt_optimization import main


class _LaunchCapturedError(Exception):
    pass


def test_run_claude_passes_strict_local_mcp_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "repo_root", lambda: tmp_path)
    launched: list[list[str]] = []

    def fake_popen(cmd: list[str], *args: object, **kwargs: object) -> None:
        launched.append(cmd)
        raise _LaunchCapturedError

    monkeypatch.setattr(main.subprocess, "Popen", fake_popen)
    args = argparse.Namespace(
        agent_spec="claim_history:train",
        model="gpt-4.1-mini-2025-04-14",
        rounds=1,
        skill="full",
        loops="",
        optimizer_model="opus",
        budget=1.0,
        max_chars=20_000,
    )

    with pytest.raises(_LaunchCapturedError):
        main.run_claude(args, "tmp_out", start_round=0)

    cmd = launched[0]
    assert cmd[0] == "claude"
    assert "--strict-mcp-config" in cmd
    config_path = Path(cmd[cmd.index("--mcp-config") + 1])
    assert config_path == tmp_path / "tmp_out" / "mcp_config.json"
    server = json.loads(config_path.read_text())["mcpServers"]["coach-light"]
    assert server == {"type": "stdio", "command": sys.executable, "args": [str(main.MCP_SERVER)]}
    assert main.MCP_SERVER.is_file()
    allowed = cmd[cmd.index("--allowedTools") + 1].split()
    assert allowed and all(tool.startswith("mcp__coach-light__") for tool in allowed)
