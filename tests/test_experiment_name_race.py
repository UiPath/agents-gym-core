"""_next_experiment_name never returns the same name twice, even to concurrent callers."""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agents_gym_core.prompt_optimization.mcp_tools import tools_runner


class TestNextExperimentNameRace:
    def test_concurrent_calls_get_different_names(self, tmp_path: Path) -> None:
        barrier = threading.Barrier(16, timeout=10)

        def reserve() -> str:
            barrier.wait()
            return tools_runner._next_experiment_name("mcp_run")

        with patch.object(tools_runner, "experiments_dir", return_value=tmp_path):
            tools_runner._reserved_names.clear()
            with ThreadPoolExecutor(16) as pool:
                names = list(pool.map(lambda _: reserve(), range(16)))

        assert sorted(names) == [f"mcp_run_{i:03d}" for i in range(1, 17)]

    def test_respects_existing_directories(self, tmp_path: Path) -> None:
        (tmp_path / "mcp_run_003").mkdir()
        with patch.object(tools_runner, "experiments_dir", return_value=tmp_path):
            tools_runner._reserved_names.clear()
            name = tools_runner._next_experiment_name("mcp_run")
            assert name == "mcp_run_004"

    def test_release_allows_reuse_scan(self, tmp_path: Path) -> None:
        with patch.object(tools_runner, "experiments_dir", return_value=tmp_path):
            tools_runner._reserved_names.clear()
            name1 = tools_runner._next_experiment_name("mcp_run")
            assert name1 == "mcp_run_001"
            tools_runner._release_experiment_name(name1)
            # Without the folder on disk, the next call reuses 001
            name2 = tools_runner._next_experiment_name("mcp_run")
            assert name2 == "mcp_run_001"
