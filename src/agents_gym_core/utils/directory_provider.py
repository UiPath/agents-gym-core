"""Provider for agent directory paths."""

from pathlib import Path

from agents_gym_core.paths import project_root


class DirectoryProvider:
    """Base class for providing agent directory paths."""

    def get_agents_dir(self) -> Path:
        """Get the agents directory path."""
        raise NotImplementedError

    def solve_path(self, agent_name: str, file_path: str) -> Path:
        raise NotImplementedError

    def check_for_placeholder(self, path: str) -> bool:
        raise NotImplementedError

    def solve_evaluators_paths(self, agent_name: str) -> list[Path]:
        raise NotImplementedError

    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the agent directory path for a given agent name."""
        agents_dir = self.get_agents_dir()
        agent_dir = agents_dir / agent_name

        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

        return agent_dir

    def _agent_exists(self, agent_name: str) -> bool:
        """Check if agent has a base config (agent.yaml) in the given directory."""
        try:
            agent_dir = self.get_agent_dir(agent_name)
            config_file = agent_dir / "agent.yaml"
            return config_file.exists()
        except FileNotFoundError:
            return False

    def _version_exists(self, agent_name: str) -> bool:
        """Check if agent has versions config (versions.yaml) in the given directory."""
        try:
            agent_dir = self.get_agent_dir(agent_name)
            versions_file = agent_dir / "versions.yaml"
            return versions_file.exists()
        except FileNotFoundError:
            return False

    def get_implicit_tag(self) -> str | None:
        """Return the implicit tag for this provider, or None if not applicable.

        When a provider has an implicit tag, all agents in that provider's
        directory are considered to have that tag without needing it in their
        agent.yaml metadata.
        """
        return None

    def get_project_root(self) -> Path:
        return project_root()

    def agent_roots(self, source: str) -> tuple[Path, ...]:
        gym = self.get_project_root() / "gym" / "agents"
        core = self.get_agents_dir()
        return (gym, core) if source == "all" else (gym if source == "gym" else core,)

    def private_data_dirs(self) -> tuple[str, ...]:
        return ("mock_data",)


class CoreDirectoryProvider(DirectoryProvider):
    """Provider for the agents bundled with agents_gym_core (``core::`` paths)."""

    def get_implicit_tag(self) -> str | None:
        return "core"

    def get_agents_dir(self) -> Path:
        """Get the agents directory path."""
        return Path(__file__).parent.parent / "agents"

    def check_for_placeholder(self, path: str):
        if not path.startswith("core::"):
            return False
        return True

    def solve_path(self, agent_name: str, file_path: str):
        if not self.check_for_placeholder(file_path):
            raise Exception("File path provided doesn't start with the expected placeholder!")
        return self.get_agent_dir(agent_name) / file_path.split("core::", maxsplit=1)[1]

    def solve_evaluators_paths(self, agent_name: str) -> list[Path]:
        return [self.get_agent_dir(agent_name) / "evaluations" / "evaluators"]
