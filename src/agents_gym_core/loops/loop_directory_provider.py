from pathlib import Path

from agents_gym_core.utils import DirectoryProvider


class LoopDirectoryProvider(DirectoryProvider):
    def get_agents_dir(self) -> Path:
        """Get the agents directory path."""
        return Path(__file__).parent

    def check_for_placeholder(self, path: str):
        if not path.startswith("core::"):
            return False
        return True

    def solve_path(self, agent_name: str, file_path: str):
        if not self.check_for_placeholder(file_path):
            raise Exception("File path provided doesn't start with the expected placeholder!")
        return self.get_agent_dir(agent_name) / file_path.split("core::", maxsplit=1)[1]

    def solve_evaluators_paths(self, agent_name: str) -> list[Path]:
        raise Exception("This method is not valid for LoopDirectoryProvider!")
