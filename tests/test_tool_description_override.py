"""A ``description:`` next to a ``@tool`` target applies to that version only."""

import os

import pytest

from agents_gym_core.hydra_agent import CoreAgentRegistry


@pytest.fixture(scope="module")
def registry() -> CoreAgentRegistry:
    os.environ.setdefault("UIPATH_GYM_SKIP_INIT", "true")
    return CoreAgentRegistry()


def _descriptions(registry: CoreAgentRegistry, version: str) -> dict[str, str]:
    return {t.name: t.description for t in registry.get_agent("calculator", version).tools}


def test_the_override_applies(registry: CoreAgentRegistry) -> None:
    assert _descriptions(registry, "bad_description_add_tool")["add"] == "Calculate this."


def test_the_shared_tool_keeps_its_description(registry: CoreAgentRegistry) -> None:
    _descriptions(registry, "bad_description_add_tool")
    assert _descriptions(registry, "base")["add"] == "Add two numbers together."
