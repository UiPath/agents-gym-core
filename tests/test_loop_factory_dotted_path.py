"""A loop given as a dotted path is imported directly: a BasicLoop subclass or a graph factory.

``LoopDetails`` records the dotted path, not a ``LoopName``.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agents_gym_core.loops.loop_factory import (
    _import_loop_class,
    get_loop,
    get_uncompiled_loop,
)


def test_a_path_without_a_module_is_rejected() -> None:
    with pytest.raises(ImportError, match="Invalid loop path"):
        _import_loop_class("JustAClassName")


# ---------------------------------------------------------------------------
# End-to-end: get_uncompiled_loop with dotted path
# ---------------------------------------------------------------------------


class _MinimalInput(BaseModel):
    question: str


class _MinimalOutput(BaseModel):
    answer: str


class _MinimalState(_MinimalInput):
    answer: str = ""


def _minimal_graph_factory(*, scenario: object, model: object) -> StateGraph:
    """Test factory with no dependency on BasicLoop or unrelated factory kwargs."""
    del scenario, model

    def answer(state: _MinimalState) -> dict[str, str]:
        del state
        return {"answer": "done"}

    builder = StateGraph(
        state_schema=_MinimalState,
        input_schema=_MinimalInput,
        output_schema=_MinimalOutput,
    )
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder


def _make_fake_llm():
    """Build a minimal LLM mock that satisfies isinstance(x, BaseChatModel)."""
    from unittest.mock import MagicMock

    from langchain_core.language_models import BaseChatModel

    llm = MagicMock(spec=BaseChatModel)
    llm.model_name = "fake-model"
    return llm


def _make_fake_agent():
    """Build a minimal Agent that satisfies BasicLoop.__init__."""
    from unittest.mock import MagicMock

    from uipath.eval.models.evaluation_set import EvaluationSet

    agent = MagicMock()
    agent.name = "test_agent"
    agent.version = "base"
    agent.system_prompt = "You are a test agent."
    agent.user_prompt = "Question: {question}"
    agent.input_schema = _MinimalInput
    agent.output_schema = _MinimalOutput
    agent.tools = []
    agent.tags = []
    agent.is_conversational = False
    agent.evaluation_set = EvaluationSet(id="test", name="test", evaluations=[])
    return agent


class TestGetUncompiledLoopDottedPath:
    """Test that get_uncompiled_loop accepts a dotted Python path."""

    def test_basic_loop_via_dotted_path(self) -> None:
        """BasicLoop resolved via dotted path builds a graph."""
        graph = get_uncompiled_loop(
            loop_name="agents_gym_core.loops.basic_loop.BasicLoopAny",
            version="any",
            scenario=_make_fake_agent(),
            models={"model": _make_fake_llm()},
        )
        # The graph should exist and have loop_details attached
        assert hasattr(graph, "loop_details")
        details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
        assert details.loop_name == "agents_gym_core.loops.basic_loop.BasicLoopAny"
        assert details.loop_version == "any"
        assert "fake-model" in details.models.get("model", "")

    @pytest.mark.asyncio
    async def test_graph_factory_compiles_through_get_loop(self) -> None:
        path = f"{__name__}._minimal_graph_factory"
        graph = get_loop(
            loop_name=path,
            version="auto",
            scenario=_make_fake_agent(),
            models={"model": _make_fake_llm()},
            print_trace=True,
        )

        details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
        assert (details.loop_name, details.loop_version) == (path, "auto")
        assert await graph.ainvoke({"question": "anything"}) == {"answer": "done"}

    def test_nonexistent_dotted_path_raises(self) -> None:
        with pytest.raises(ImportError):
            get_uncompiled_loop(
                loop_name="nonexistent.module.FakeLoop",
                version="auto",
                scenario=_make_fake_agent(),
                models={"model": _make_fake_llm()},
            )

    def test_non_basicloop_class_raises_type_error(self) -> None:
        """A dotted path that resolves to neither supported target should fail."""
        with pytest.raises(TypeError, match="neither a BasicLoop subclass nor a graph factory"):
            get_uncompiled_loop(
                loop_name="agents_gym_core.loops.types.LoopName",
                version="auto",
                scenario=_make_fake_agent(),
                models={"model": _make_fake_llm()},
            )

    def test_unknown_named_loop_still_raises_value_error(self) -> None:
        """Short names not in LoopName enum still fail with ValueError."""
        with pytest.raises(ValueError, match="is not a valid LoopName"):
            get_uncompiled_loop(
                loop_name="nonexistent-loop",
                version="auto",
                scenario=_make_fake_agent(),
                models={"model": _make_fake_llm()},
            )


# ---------------------------------------------------------------------------
# LoopDetails with string loop_name
# ---------------------------------------------------------------------------


class TestLoopDetailsWithStringName:
    def test_loop_key_with_dotted_path(self) -> None:
        from agents_gym_core.loops.types import LoopDetails

        details = LoopDetails(
            loop_name="gym.experiments.my_loop.MyLoop",
            loop_version="auto",
            models={"model": "gpt-4o"},
        )
        assert details.loop_key == "gym.experiments.my_loop.MyLoop__auto__model:gpt-4o"
