"""An explicitly selected standalone project's directory provider."""

import json
import re
from pathlib import Path

from agents_gym_core.utils.directory_provider import CoreDirectoryProvider

PROJECT_FILE = "gym-project.json"


class ProjectDirectoryProvider(CoreDirectoryProvider):
    """Resolve one project once; registries and optimizer workers share its policy.

    No environment activation, parent search, or Python import-path mutation. Workers
    receive an explicit CLI flag and resolve their own provider once.
    Optimizer/server entry points require this exact class: workers reconstruct it
    from the root rather than carrying custom provider implementations.
    """

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        config = json.loads((self.root / PROJECT_FILE).read_text())
        if not isinstance(config, dict) or config.get("version") != 1:
            raise ValueError(f"{self.root / PROJECT_FILE}: expected an object with version: 1")

    def get_project_root(self) -> Path:
        return self.root

    def get_agents_dir(self) -> Path:
        return self.root / "agents"

    def get_implicit_tag(self) -> str | None:
        return "project"

    def agent_roots(self, source: str) -> tuple[Path, ...]:
        return (self.get_agents_dir(),)

    def private_data_dirs(self) -> tuple[str, ...]:
        return ("mock_data", "data")

    def get_agent_dir(self, agent_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", agent_name):
            raise ValueError(f"Invalid project agent name: {agent_name!r}")
        path = super().get_agent_dir(agent_name).resolve()
        if not path.is_relative_to(self.get_agents_dir().resolve()):
            raise ValueError(f"Project agent directory escapes agents/: {agent_name}")
        return path
