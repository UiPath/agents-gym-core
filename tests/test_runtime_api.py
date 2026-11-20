"""The Python API runs real Gym loops without a provider connection."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from agents_gym_core.runtime.api import GymRuntime


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolModel":
        return self


def calculator() -> GymRuntime:
    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "multiply", "args": {"a": 17, "b": 23}, "id": "multiply-1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_execution", "args": {"answer": 391}, "id": "end-1"}],
            ),
        ]
    )
    return GymRuntime(model)


async def test_run_calculator() -> None:
    gym = calculator()
    assert await gym.run("calculator", {"expression": "17 * 23"}) == {"answer": 391}


async def test_deep_agents_run_uses_scenario_inputs_and_returns_normalized_output() -> None:
    from langchain_core.messages import HumanMessage

    seen: list[Any] = []

    class RecordingModel(ToolModel):
        async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            seen.extend(messages)
            return await super()._agenerate(messages, *args, **kwargs)

    gym = GymRuntime(
        RecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "end_execution", "args": {"answer": 391}, "id": "end-1"}],
                )
            ]
        ),
        loop_name="deep-agents",
    )
    assert await gym.run("calculator", {"expression": "17 * 23"}) == {"answer": 391}
    assert any(isinstance(msg, HumanMessage) and "17 * 23" in msg.content for msg in seen)
    seen.clear()
    with pytest.raises(ValidationError, match="expression"):
        await gym.run("calculator", {})
    assert not seen


async def test_deep_agents_run_normalizes_raise_error() -> None:
    gym = GymRuntime(
        ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "raise_error",
                            "args": {"message": "Unable to calculate", "details": "missing data"},
                            "id": "error-1",
                        }
                    ],
                )
            ]
        ),
        loop_name="deep-agents",
    )
    result = await gym.run("calculator", {"expression": "17 * 23"})
    assert result["type"] == "error"
    assert result["title"] == "Unable to calculate"
    assert result["details"] == "missing data"
    assert "TERMINATION_LLM_RAISED_ERROR" in result["code"]


async def test_invalid_input_is_rejected() -> None:
    gym = calculator()
    with pytest.raises(ValidationError):
        await gym.run("calculator", {})


async def test_native_graph_streams_tool_execution() -> None:
    graph = calculator().graph("calculator")
    events = [
        event async for event in graph.astream_events({"expression": "17 * 23"}, version="v2")
    ]
    tools = [event["name"] for event in events if event["event"] == "on_tool_start"]
    assert "multiply" in tools
    assert any(event["event"] == "on_tool_end" and event["name"] == "multiply" for event in events)
    data = events[-1]["data"]
    assert "output" in data
    assert data["output"]["answer"] == 391


async def test_run_honors_graph_recursion_limit() -> None:
    with pytest.raises(GraphRecursionError):
        await calculator().run(
            "calculator", {"expression": "17 * 23"}, config={"recursion_limit": 1}
        )


async def test_agent_and_loop_versions() -> None:
    gym = calculator()
    gym.loop_version = "any"
    assert await gym.run("calculator", {"expression": "17 * 23"}, version="worst_sys_prompt") == {
        "answer": 391
    }


async def test_custom_registry_tools_receive_runtime_embeddings() -> None:
    from langchain_core.embeddings import Embeddings, FakeEmbeddings
    from langchain_core.tools import BaseTool

    from agents_gym_core.hydra_agent import CoreAgentRegistry
    from agents_gym_core.utils.gym_types import Agent

    embedder = FakeEmbeddings(size=3)

    class SearchTool(BaseTool):
        name: str = "search"
        description: str = "Search for an answer."
        embeddings: Embeddings | None = None
        embeddings_model_name: str | None = None

        def _run(self, query: str) -> str:
            assert self.embeddings is embedder
            return "391"

    class Registry(CoreAgentRegistry):
        def get_agent(
            self,
            agent_name: str,
            agent_version: str = "base",
            *,
            tools: list[BaseTool] | None = None,
        ) -> Agent:
            agent = super().get_agent(agent_name, agent_version, tools=tools)
            agent.tools = [SearchTool()]
            return agent

    model = ToolModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "search", "args": {"query": "answer"}, "id": "s"}]
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_execution", "args": {"answer": 391}, "id": "e"}],
            ),
        ]
    )
    gym = GymRuntime(model, registry=Registry(), embeddings=embedder)
    assert await gym.run("calculator", {"expression": "17 * 23"}) == {"answer": 391}


@pytest.mark.parametrize(
    ("agent_loop", "loop_version", "expected"),
    [
        ("dspy-program:pkg.Program", "auto", ("dspy-program", "pkg.Program")),
        ("basic:any", "auto", ("basic", "any")),
        ("", "auto", ("basic", "auto")),
        ("dspy-program:pkg.Program", "any", ("basic", "any")),  # an explicit variant wins
    ],
)
def test_graph_uses_the_agents_own_loop_as_the_testbed_does(
    monkeypatch: pytest.MonkeyPatch, agent_loop: str, loop_version: str, expected: tuple
) -> None:
    from agents_gym_core.hydra_agent import CoreAgentRegistry
    from agents_gym_core.runtime import api
    from agents_gym_core.utils.gym_types import Agent

    class Registry(CoreAgentRegistry):
        def get_agent(
            self,
            agent_name: str,
            agent_version: str = "base",
            *,
            tools: list[BaseTool] | None = None,
        ) -> Agent:
            agent = super().get_agent(agent_name, agent_version, tools=tools)
            agent.loop = agent_loop
            return agent

    built: list[tuple[str, str]] = []
    monkeypatch.setattr(
        api,
        "get_loop",
        lambda name, version, *a, **k: built.append((name, version)) or SimpleNamespace(),
    )
    GymRuntime(ToolModel(responses=[]), registry=Registry(), loop_version=loop_version).graph(
        "calculator"
    )
    assert built == [expected]


def test_tool_override_skips_default_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents_gym_core.hydra_agent import CoreAgentRegistry
    from agents_gym_core.hydra_agent.tool_instantiator import ToolInstantiator

    def forbidden(_config: Any) -> None:
        raise AssertionError("Default tool factory must not run")

    monkeypatch.setattr(ToolInstantiator, "instantiate_tools", forbidden)
    registry = CoreAgentRegistry(load_evaluations=False)
    agent = registry.get_agent("sql", tools=[])
    assert agent.name == "sql" and agent.tools == []
    assert agent.system_prompt


def test_runtime_does_not_load_evaluation_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader

    def forbidden(*args: Any) -> None:
        raise AssertionError("Single-agent execution does not need evaluation datasets")

    monkeypatch.setattr(EvaluationLoader, "load_evaluation_set", forbidden)
    agent = calculator().registry.get_agent("calculator")
    assert agent.name == "calculator"


def test_explicit_input_schema_replaces_only_that_graph() -> None:
    from pydantic import BaseModel

    class Question(BaseModel):
        CustomerQuery: str

    gym = calculator()
    custom = gym.graph("product_questions_agent", tools=[], input_schema=Question)
    assert set(custom.get_input_schema().model_fields) == {"CustomerQuery"}
    original = gym.registry.get_agent("product_questions_agent", tools=[])
    assert set(original.input_schema.model_fields) == {
        "CustomerQuery",
        "ProductName",
        "ProductManufacturer",
    }


async def test_run_uses_explicit_tools_and_keeps_other_loads_unchanged() -> None:
    from langchain_core.tools import tool

    @tool
    def answer() -> int:
        """Return a test value."""
        return 391

    gym = calculator()
    gym.model = ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "answer", "args": {}, "id": "custom"}]),
            AIMessage(
                content="",
                tool_calls=[{"name": "end_execution", "args": {"answer": 391}, "id": "end"}],
            ),
        ]
    )
    assert await gym.run("calculator", {"expression": "test"}, tools=[answer]) == {"answer": 391}
    assert "answer" not in [t.name for t in gym.registry.get_agent("calculator").tools]


def test_skipping_evaluations_preserves_agent_initialization_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents_gym_core.hydra_agent import CoreAgentRegistry
    from agents_gym_core.hydra_agent.agent_init import AgentInit

    checked = []
    monkeypatch.setattr(AgentInit, "has_init", lambda cfg: True)
    monkeypatch.setattr(AgentInit, "check_or_raise", lambda cfg, name, *args: checked.append(name))
    CoreAgentRegistry(load_evaluations=False).get_agent("calculator")
    assert checked == ["calculator"]


def test_registry_subclass_without_super_keeps_loading_evaluations() -> None:
    from agents_gym_core.hydra_agent import CoreAgentRegistry
    from agents_gym_core.utils.directory_provider import CoreDirectoryProvider

    class Registry(CoreAgentRegistry):
        def __init__(self):
            self.core_directory_provider = CoreDirectoryProvider()

    assert Registry().get_agent("calculator").evaluation_set.evaluations


async def test_evaluate_uses_native_scoring_and_saves_reports(tmp_path: Any) -> None:
    import json

    from uipath.eval.evaluators import BaseEvaluator
    from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
    from uipath.eval.models.evaluation_set import EvaluationItem

    checks = [
        EvaluatorFactory.create_evaluator(
            {
                "version": "1.0",
                "id": name,
                "description": name,
                "evaluatorTypeId": kind,
                "evaluatorConfig": settings,
            }
        )
        for name, kind, settings in [
            ("exact", "uipath-exact-match", {"targetOutputKey": "*", "caseSensitive": True}),
            ("count", "uipath-tool-call-count", {"strict": True}),
            ("order", "uipath-tool-call-order", {"strict": True}),
        ]
    ]
    case = EvaluationItem(
        id="math",
        name="math",
        inputs={"expression": "17 * 23"},
        evaluationCriterias={
            "exact": {"expectedOutput": {"answer": 391}},
            "count": {"toolCallsCount": {"multiply": ["=", 1]}},
            "order": {"toolCallsOrder": ["multiply"]},
        },
    )
    assert all(isinstance(check, BaseEvaluator) for check in checks)
    from typing import cast

    report = await calculator().evaluate(
        "calculator",
        [case],
        cast(list[BaseEvaluator], checks),
        repeats=2,
        output_dir=tmp_path / "evaluation",
    )
    results = [
        entry
        for loops in report["evaluation_results"].values()
        for evaluators in loops.values()
        for cases in evaluators.values()
        for entry in cases.values()
    ]
    assert len(results) == 6
    assert all(entry["result"].score == 1.0 for entry in results)
    assert {entry["run_number"] for entry in results} == {1, 2}
    assert all(entry["agent_output"] == {"answer": 391} for entry in results)
    assert all(path.exists() for path in report["saved_files"].values())
    assert len(json.loads(report["saved_files"]["combined_results"].read_text())) == 6
    assert len(list((tmp_path / "evaluation").rglob("output.json"))) == 2
    assert report["failed_datapoint_runs"] == []
    assert report["evaluation_complete"] is True

    case.evaluation_criterias["count"] = {"toolCallsCount": {"multiply": ["=", 2]}}
    weighted = await calculator().evaluate(
        "calculator",
        [case],
        cast(list[BaseEvaluator], checks),
        evaluator_weights={"exact": 1, "count": 0, "order": 0},
        output_dir=tmp_path / "weighted",
    )
    assert weighted["metrics_df"]["count"].iloc[0] == 0.0
    assert all(
        score.runs_mean == 1.0
        for agents in weighted["aggregated_scores"].values()
        for splits in agents.values()
        for score in splits.values()
    )


async def test_evaluate_rejects_missing_evaluators_before_running() -> None:
    from uipath.eval.models.evaluation_set import EvaluationItem

    case = EvaluationItem(
        id="math",
        name="math",
        inputs={"expression": "17 * 23"},
        evaluationCriterias={"missing": {}},
    )
    with pytest.raises(ValueError, match="Unknown evaluator IDs"):
        await calculator().evaluate("calculator", [case], [])


async def test_evaluate_reports_cases_excluded_from_aggregates(tmp_path: Any) -> None:
    from uipath.eval.evaluators import BaseEvaluator
    from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
    from uipath.eval.models.evaluation_set import EvaluationItem

    check = EvaluatorFactory.create_evaluator(
        {
            "version": "1.0",
            "id": "exact",
            "description": "exact",
            "evaluatorTypeId": "uipath-exact-match",
            "evaluatorConfig": {"targetOutputKey": "*", "caseSensitive": True},
        }
    )
    assert isinstance(check, BaseEvaluator)
    cases = [
        EvaluationItem(
            id=name,
            name=name,
            inputs={"expression": "17 * 23"},
            evaluationCriterias={"exact": criteria},
        )
        for name, criteria in [
            ("valid", {"expectedOutput": {"answer": 391}}),
            ("invalid", {"wrongKey": {}}),
        ]
    ]
    with pytest.warns(RuntimeWarning, match="1 of 2 datapoint runs failed"):
        report = await calculator().evaluate(
            "calculator", cases, [check], output_dir=tmp_path / "incomplete"
        )
    assert report["evaluation_complete"] is False
    assert len(report["failed_datapoint_runs"]) == 1
    assert "invalid run 1:" in report["failed_datapoint_runs"][0]
    assert "valid:run_1" in next(
        cases
        for loops in report["evaluation_results"].values()
        for evaluators in loops.values()
        for cases in evaluators.values()
    )


def test_reloading_tracing_does_not_wrap_the_encoder_recursively() -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
from agents_gym_core.utils import trace_utils
for _ in range(3):
    importlib.reload(trace_utils)
    assert trace_utils._otlp_internal._encode_value('ok').string_value == 'ok'
    assert trace_utils._otlp_internal._encode_value(7).int_value == 7
    assert trace_utils._otlp_internal._encode_value(2**100).string_value.startswith('0x')
""",
        ],
        check=True,
    )


async def test_evaluate_scores_parallel_tools_by_start_order(tmp_path: Any) -> None:
    import asyncio
    from typing import TypedDict

    from langgraph.graph import START, StateGraph
    from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
    from uipath.eval.models.evaluation_set import EvaluationItem

    from agents_gym_core.utils.trace_utils import setup_tracer

    tracing = setup_tracer()
    tracer = tracing[1].get_tracer("parallel-test")

    class State(TypedDict, total=False):
        answer: int

    async def worker(state: State) -> State:
        first_started, second_finished = asyncio.Event(), asyncio.Event()

        async def first() -> None:
            with tracer.start_as_current_span("first", attributes={"tool.name": "first"}):
                first_started.set()
                await second_finished.wait()

        async def second() -> None:
            await first_started.wait()
            with tracer.start_as_current_span("second", attributes={"tool.name": "second"}):
                second_finished.set()

        await asyncio.gather(first(), second())
        return {"answer": 1}

    builder = StateGraph(State)
    builder.add_node("worker", worker)
    builder.add_edge(START, "worker")
    graph = builder.compile()
    gym = calculator()
    original = gym.graph("calculator")
    graph.loop_details = original.loop_details  # pyright: ignore[reportAttributeAccessIssue]
    setattr(graph, "_gym_agent", getattr(original, "_gym_agent"))
    check = EvaluatorFactory.create_evaluator(
        {
            "version": "1.0",
            "id": "order",
            "description": "order",
            "evaluatorTypeId": "uipath-tool-call-order",
            "evaluatorConfig": {"strict": True},
        }
    )
    from uipath.eval.evaluators import BaseEvaluator

    assert isinstance(check, BaseEvaluator)
    case = EvaluationItem(
        id="parallel",
        name="parallel",
        inputs={},
        evaluationCriterias={"order": {"toolCallsOrder": ["first", "second"]}},
    )
    report = await gym.evaluate(
        "calculator",
        [case],
        [check],
        graph=graph,
        tracing=tracing,
        output_dir=tmp_path / "parallel",
    )
    entries = [
        entry
        for loops in report["evaluation_results"].values()
        for evaluators in loops.values()
        for cases in evaluators.values()
        for entry in cases.values()
    ]
    assert entries[0]["result"].score == 1.0


async def test_compare_basic_and_planning_in_one_experiment(tmp_path: Any) -> None:
    from uipath.eval.evaluators import BaseEvaluator
    from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
    from uipath.eval.models.evaluation_set import EvaluationItem

    basic = calculator()
    assert isinstance(basic.model, ToolModel)
    from langchain_core.messages.ai import UsageMetadata

    usage: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    for message in basic.model.responses:
        assert isinstance(message, AIMessage)
        message.usage_metadata = usage
    planning_model = ToolModel(
        responses=[
            AIMessage(content="Multiply 17 by 23.", usage_metadata=usage),
            *basic.model.responses,
        ]
    )
    planning = GymRuntime(planning_model, loop_name="planning")
    check = EvaluatorFactory.create_evaluator(
        {
            "version": "1.0",
            "id": "exact",
            "description": "exact",
            "evaluatorTypeId": "uipath-exact-match",
            "evaluatorConfig": {"targetOutputKey": "*"},
        }
    )
    assert isinstance(check, BaseEvaluator)
    case = EvaluationItem(
        id="math",
        name="math",
        inputs={"expression": "17 * 23"},
        evaluationCriterias={"exact": {"expectedOutput": {"answer": 391}}},
    )
    report = await basic.evaluate(
        "calculator",
        [case],
        [check],
        graphs=[basic.graph("calculator"), planning.graph("calculator")],
        concurrency=2,
        output_dir=tmp_path / "comparison",
    )
    assert set(report["metrics_df"]["loop_name"]) == {"basic", "planning"}
    assert report["metrics_df"]["exact"].tolist() == [1.0, 1.0]
    assert sorted(report["costs_df"]["num_llm_calls"].tolist()) == [2, 3]
    assert set(report["costs_df"]["loop_name"]) == {"basic", "planning"}
    assert len(report["saved_files"]) == 4


async def test_native_judge_accepts_model_without_loading_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from agents_gym_core.models.judge_routing import routed_llm_service

    monkeypatch.setitem(sys.modules, "agents_gym_core.models.chat.clients", None)
    response = await routed_llm_service(
        [{"role": "user", "content": "Check the answer."}],
        chat_model=ToolModel(responses=[AIMessage(content="ok")]),
    )
    assert response.choices[0].message.content == "ok"


async def test_evaluate_binds_judge_model_without_mutating_evaluator(tmp_path: Any) -> None:
    from uipath.eval.evaluators import BaseEvaluator
    from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
    from uipath.eval.models.evaluation_set import EvaluationItem

    class JudgeModel(ToolModel):
        model_name: str = "explicit-judge"

    judge_model = JudgeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_evaluation",
                        "id": "judge-1",
                        "args": {"score": 100, "justification": "Both answers are 391."},
                    }
                ],
            )
        ]
    )
    evaluator = EvaluatorFactory.create_evaluator(
        {
            "version": "1.0",
            "id": "judge",
            "description": "judge",
            "evaluatorTypeId": "uipath-llm-judge-output-semantic-similarity",
            "evaluatorConfig": {"model": "original-model", "targetOutputKey": "*"},
        }
    )
    assert isinstance(evaluator, BaseEvaluator)
    original_service = getattr(evaluator, "llm_service")
    case = EvaluationItem(
        id="math",
        name="math",
        inputs={"expression": "17 * 23"},
        evaluationCriterias={"judge": {"expectedOutput": {"answer": 391}}},
    )
    report = await calculator().evaluate(
        "calculator", [case], [evaluator], judge_model=judge_model, output_dir=tmp_path / "judged"
    )
    entries = [
        entry
        for loops in report["evaluation_results"].values()
        for evaluators in loops.values()
        for cases in evaluators.values()
        for entry in cases.values()
    ]
    assert entries[0]["result"].score == 1.0
    assert entries[0]["judge_model"] == "explicit-judge"
    assert getattr(evaluator.evaluator_config, "model") == "original-model"
    assert getattr(evaluator, "llm_service") is original_service
