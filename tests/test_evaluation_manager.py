"""EvaluationManager builds an agent's evaluators once and reuses them for every datapoint."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.runtime.runners.evaluation_manager import EvaluationManager
from agents_gym_core.runtime.runtime_config import RuntimeConfig


def _manager() -> EvaluationManager:
    return EvaluationManager(registry=cast(CoreAgentRegistry, MagicMock()), config=RuntimeConfig())


def test_get_evaluators_memoized_per_agent() -> None:
    mgr = _manager()
    with patch(
        "agents_gym_core.runtime.runners.evaluation_manager.EvaluationLoader.get_evaluators",
        side_effect=lambda agent_name, directory_provider: {agent_name: cast(Any, object())},
    ) as loader:
        first = mgr._get_evaluators("agent_a")
        again = mgr._get_evaluators("agent_a")

        # Same object reused; the expensive loader ran exactly once.
        assert first is again
        assert loader.call_count == 1

        # A different agent gets its own build (correctness across agents).
        other = mgr._get_evaluators("agent_b")
        assert other is not first
        assert loader.call_count == 2

        # Repeated calls never rebuild.
        for _ in range(100):
            mgr._get_evaluators("agent_a")
            mgr._get_evaluators("agent_b")
        assert loader.call_count == 2
