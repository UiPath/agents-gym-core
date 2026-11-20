"""Loop versions are named after their behaviour.

``basic`` is ``basic:auto``. ``any`` forces a tool call every turn, and ``-graceful`` forces
``end_execution`` at the call limit instead of failing.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.loops import (
    BasicLoop,
    BasicLoopAny,
    BasicLoopAnyGraceful,
    BasicLoopAutoGraceful,
    SummarizingLoop,
    SummarizingLoopAny,
)
from agents_gym_core.loops.loop_factory import (
    _check_deep_agents_version,
    _create_loop_instance,
    get_default_graph_config,
)
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION, LoopName
from agents_gym_core.testbed_core import TestbedRunnerCore
from agents_gym_core.utils.gym_settings_core import gym_settings_core


class _Input(BaseModel):
    question: str


class _Output(BaseModel):
    answer: str


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.name = "test_agent"
    agent.system_prompt = "You are a test agent."
    agent.user_prompt = "Question: {question}"
    agent.input_schema = _Input
    agent.output_schema = _Output
    agent.tools = []
    agent.tags = []
    agent.is_conversational = False
    agent.evaluation_set = EvaluationSet(id="test", name="test", evaluations=[])
    return agent


def _model() -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.model_name = "fake-model"
    return model


def _loop(loop_name: str, version: str) -> BasicLoop:
    model = _model()
    return _create_loop_instance(
        loop_name,
        version,
        _agent(),
        model,
        {"model": model},
        get_default_graph_config(gym_settings_core),
        gym_settings_core,
    )


@pytest.mark.parametrize(
    ("loop_name", "version", "cls", "tool_choice", "graceful", "repetition"),
    [
        ("basic", "auto", BasicLoop, "auto", False, False),
        ("basic", "any", BasicLoopAny, "any", False, False),
        ("basic", "auto-graceful", BasicLoopAutoGraceful, "auto", True, True),
        ("basic", "any-graceful", BasicLoopAnyGraceful, "any", True, True),
        # The summarizing loop always ends gracefully at the call limit.
        ("summarizing", "auto", SummarizingLoop, "auto", True, False),
        ("summarizing", "any", SummarizingLoopAny, "any", True, False),
    ],
)
def test_each_version_runs_its_loop(
    loop_name: str, version: str, cls: type, tool_choice: str, graceful: bool, repetition: bool
) -> None:
    loop = _loop(loop_name, version)

    assert type(loop) is cls
    assert loop.tool_choice == tool_choice
    assert loop.graceful_termination is graceful
    assert loop.detect_tool_repetition is repetition
    assert loop.details.loop_version == version


def test_planning_and_validation_default_to_auto() -> None:
    for loop_name in (LoopName.PLANNING, LoopName.VALIDATION):
        assert _loop(loop_name, DEFAULT_LOOP_VERSION).details.loop_version == "auto"


def test_deep_agents_has_one_version() -> None:
    _check_deep_agents_version("auto")
    with pytest.raises(ValueError, match="Unknown loop: deep-agents:any"):
        _check_deep_agents_version("any")


def test_a_loop_without_a_version_is_auto() -> None:
    runner = TestbedRunnerCore.__new__(TestbedRunnerCore)

    assert runner._parse_loops(["basic", "planning", "basic:any"]) == [
        ("basic", "auto"),
        ("planning", "auto"),
        ("basic", "any"),
    ]


def test_model_roles_reach_the_loop_by_name() -> None:
    model, planner = _model(), _model()
    planner.model_name = "planner-model"
    loop = _create_loop_instance(
        "planning",
        "auto",
        _agent(),
        model,
        {"model": model, "planning_llm": planner, "untrusted_model": _model()},
        get_default_graph_config(gym_settings_core),
        gym_settings_core,
    )

    assert loop.loop_key == "planning__auto__model:fake-model,planning_llm:planner-model"
