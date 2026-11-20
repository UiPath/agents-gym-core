"""Path containment checks for MCP tools that read/write agent files."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agents_gym_core.prompt_optimization.mcp_tools.tools_agents import (
    read_agent_file,
    update_prompt,
)


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal agent directory."""
    d = tmp_path / "calculator"
    d.mkdir()
    (d / "prompts").mkdir()
    (d / "prompts" / "system_prompt.md").write_text("You are a calculator.")
    (d / "agent.yaml").write_text("version: base\n")
    return d


class TestReadAgentFileContainment:
    """read_agent_file must refuse paths that escape the agent directory."""

    def test_absolute_path_refused(self, agent_dir: Path) -> None:
        with patch(
            "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
            return_value=(agent_dir, None),
        ):
            result = read_agent_file("calculator", "/etc/passwd")
            assert "Refused" in result or "Invalid" in result

    def test_dotdot_refused(self, agent_dir: Path) -> None:
        with patch(
            "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
            return_value=(agent_dir, None),
        ):
            result = read_agent_file("calculator", "../../../etc/passwd")
            assert "Invalid" in result

    def test_symlink_escape_refused(self, agent_dir: Path) -> None:
        escape_target = agent_dir.parent / "secret.txt"
        escape_target.write_text("secret content")
        symlink = agent_dir / "escape_link"
        symlink.symlink_to(agent_dir.parent)

        with patch(
            "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
            return_value=(agent_dir, None),
        ):
            result = read_agent_file("calculator", "escape_link/secret.txt")
            assert "Refused" in result

    def test_normal_read_allowed(self, agent_dir: Path) -> None:
        with patch(
            "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
            return_value=(agent_dir, None),
        ):
            result = read_agent_file("calculator", "prompts/system_prompt.md")
            assert "You are a calculator" in result


class TestUpdatePromptContainment:
    """update_prompt must refuse writes that escape the agent directory."""

    def test_prompt_ref_outside_agent_dir_refused(self, agent_dir: Path) -> None:
        outside_file = agent_dir.parent / "outside.md"
        outside_file.write_text("original")

        with (
            patch(
                "agents_gym_core.prompt_optimization.mcp_tools.tools_agents._resolve_prompt",
                return_value=(str(outside_file), "original", None),
            ),
            patch(
                "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
                return_value=(agent_dir, None),
            ),
        ):
            result = update_prompt("calculator", "malicious content", prompt_type="system")
            assert "Refused" in result
            assert outside_file.read_text() == "original"

    def test_normal_update_allowed(self, agent_dir: Path) -> None:
        prompt_path = agent_dir / "prompts" / "system_prompt.md"

        with (
            patch(
                "agents_gym_core.prompt_optimization.mcp_tools.tools_agents._resolve_prompt",
                return_value=(str(prompt_path), "You are a calculator.", None),
            ),
            patch(
                "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.require_agent_dir",
                return_value=(agent_dir, None),
            ),
            patch(
                "agents_gym_core.prompt_optimization.mcp_tools.tools_agents.record_prompt_update",
            ),
        ):
            result = update_prompt("calculator", "Updated prompt.", prompt_type="system")
            assert "Updated" in result
            assert prompt_path.read_text().strip() == "Updated prompt."
