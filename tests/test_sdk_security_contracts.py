"""Offline contracts for the retained SDK family, using real serializers and graphs."""

from __future__ import annotations

import json
import os
import socket
import sys
import textwrap
import tomllib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import anyio
import httpx
import pytest
import pytest_asyncio
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("SDK contract fixtures must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


@pytest.mark.parametrize(
    "name",
    [
        "httpx2",
        "httpcore2",
        "uipath",
        "uipath-core",
        "uipath-langchain",
        "uipath-langchain-client",
        "uipath-llm-client",
        "uipath-platform",
        "uipath-runtime",
        "deepagents",
        "langchain",
        "langchain-core",
        "langchain-google-genai",
        "langsmith",
        "langgraph",
        "langgraph-sdk",
        "langgraph-checkpoint",
        "langgraph-checkpoint-sqlite",
        "langgraph-prebuilt",
        "mcp",
        "mcp-types",
    ],
)
def test_executed_sdk_family_matches_core_lock(name: str) -> None:
    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
    expected = {package["version"] for package in lock["package"] if package["name"] == name}
    assert expected, f"{name} is missing from the Core lock"
    assert version(name) in expected


@pytest.mark.parametrize("resource", ["threads", "assistants", "crons"])
def test_sdk_auth_resource_decorator_preserves_action_scope(resource: str) -> None:
    from langgraph_sdk.auth import Auth

    auth = Auth()

    async def allow_create(ctx: Any, value: Any) -> None:
        pass

    getattr(auth.on, resource)(actions=["create"])(allow_create)
    assert auth._handlers == {(resource, "create"): [allow_create]}


def test_sdk_sync_stream_identifiers_are_encoded() -> None:
    from langgraph_sdk._sync.stream import SyncThreadStream

    http = Mock()
    stream = SyncThreadStream(
        http=http, thread_id="../store/items?key=synthetic", assistant_id="../synthetic"
    )
    stream.agent.get_tree()
    stream._fetch_state()
    assert [call.args[0] for call in http.get.call_args_list] == [
        "/assistants/..%2Fsynthetic/graph",
        "/threads/..%2Fstore%2Fitems%3Fkey%3Dsynthetic/state",
    ]


@pytest.mark.asyncio
async def test_sdk_async_stream_identifiers_are_encoded() -> None:
    from langgraph_sdk._async.stream import AsyncThreadStream

    http = Mock(get=AsyncMock())
    stream = AsyncThreadStream(
        http=http, thread_id="../store/items?key=synthetic", assistant_id="../synthetic"
    )
    await stream.agent.get_tree()
    await stream._fetch_state()
    assert [call.args[0] for call in http.get.await_args_list] == [
        "/assistants/..%2Fsynthetic/graph",
        "/threads/..%2Fstore%2Fitems%3Fkey%3Dsynthetic/state",
    ]


@dataclass
class _Gateway:
    settings: Any
    requests: list[httpx.Request] = field(default_factory=list)
    response: dict[str, Any] = field(default_factory=dict)


@pytest_asyncio.fixture
async def gateway(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Gateway]:
    from uipath.llm_client.httpx_client import UiPathHttpxAsyncClient, UiPathHttpxClient
    from uipath_langchain_client import base_client
    from uipath_langchain_client.settings import LLMGatewaySettings

    assert LLMGatewaySettings.model_config.get("env_file") is None
    settings = LLMGatewaySettings.model_validate(
        {
            "LLMGW_URL": "https://gateway.test",
            "LLMGW_ACCESS_TOKEN": "synthetic-noncredential",
            "LLMGW_SEMANTIC_ORG_ID": "fixture-org",
            "LLMGW_SEMANTIC_TENANT_ID": "fixture-tenant",
            "LLMGW_REQUESTING_PRODUCT": "fixture",
            "LLMGW_REQUESTING_FEATURE": "sdk-upgrade",
        }
    )
    assert settings.base_url == "https://gateway.test"
    assert settings.org_id == "fixture-org" and settings.tenant_id == "fixture-tenant"
    state = _Gateway(settings=settings)
    sync_clients: list[UiPathHttpxClient] = []
    async_clients: list[UiPathHttpxAsyncClient] = []

    def respond(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        return httpx.Response(200, json=state.response)

    def sync_factory(**kwargs: Any) -> UiPathHttpxClient:
        client = UiPathHttpxClient(
            **kwargs, transport=httpx.MockTransport(respond), trust_env=False
        )
        sync_clients.append(client)
        return client

    def async_factory(**kwargs: Any) -> UiPathHttpxAsyncClient:
        client = UiPathHttpxAsyncClient(
            **kwargs, transport=httpx.MockTransport(respond), trust_env=False
        )
        async_clients.append(client)
        return client

    monkeypatch.setattr(base_client, "UiPathHttpxClient", sync_factory)
    monkeypatch.setattr(base_client, "UiPathHttpxAsyncClient", async_factory)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-noncredential")
    try:
        yield state
    finally:
        for client in sync_clients:
            client.close()
        for client in async_clients:
            await client.aclose()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "bedrock"])
@pytest.mark.asyncio
async def test_provider_request_and_usage_serialization(gateway: _Gateway, provider: str) -> None:
    from uipath_langchain_client.clients.anthropic.chat_models import UiPathChatAnthropic
    from uipath_langchain_client.clients.bedrock.chat_models import UiPathChatAnthropicBedrock
    from uipath_langchain_client.clients.google.chat_models import UiPathChatGoogleGenerativeAI
    from uipath_langchain_client.clients.openai.chat_models import UiPathAzureChatOpenAI
    from uipath_langchain_client.settings import ApiFlavor

    kwargs: dict[str, Any] = {"settings": gateway.settings, "model_details": {}, "max_retries": 0}
    if provider == "openai":
        model = UiPathAzureChatOpenAI(
            model="gpt-4o", api_flavor=ApiFlavor.CHAT_COMPLETIONS, **kwargs
        )
        gateway.response = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "fixture reply"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    elif provider == "google":
        model = UiPathChatGoogleGenerativeAI(model="gemini-2.5-flash", **kwargs)
        gateway.response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "fixture reply"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
            "modelVersion": "gemini-2.5-flash",
        }
    else:
        if provider == "anthropic":
            model = UiPathChatAnthropic(model="claude-sonnet-4-6", **kwargs)
        else:
            model = UiPathChatAnthropicBedrock(model="anthropic.claude-sonnet-4-6", **kwargs)
        # Bedrock's retained adapter uses Anthropic's raw API, not Converse.
        gateway.response = {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6"
            if provider == "anthropic"
            else "anthropic.claude-sonnet-4-6",
            "content": [{"type": "text", "text": "fixture reply"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    result = await model.ainvoke([HumanMessage(content="synthetic prompt")])

    assert "fixture reply" in str(result.content)
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] == 5
    assert result.usage_metadata["output_tokens"] == 2
    assert gateway.requests
    request = gateway.requests[-1]
    assert request.method == "POST" and request.url.host == "gateway.test"
    payload = json.loads(request.content)
    assert "synthetic prompt" in request.content.decode()
    assert "contents" in payload if provider == "google" else "messages" in payload


@pytest.mark.asyncio
async def test_mcp_stdio_schema_and_uipath_tool_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client import stdio
    from uipath_langchain.agent.tools.mcp import load_mcp_tools

    server = textwrap.dedent('''\
        import socket
        import sys
        sys.dont_write_bytecode = True
        def reject(*args, **kwargs):
            raise AssertionError("stdio fixture must not use network")
        socket.socket.connect = reject
        from mcp.server.mcpserver import MCPServer
        server = MCPServer("synthetic-stdio")
        @server.tool()
        def fixture_echo(value: str) -> str:
            """Echo a synthetic fixture value."""
            return "fixture:" + value
        server.run(transport="stdio")
        ''')
    # Do not forward the optimizer's workspace, provider keys, or home directory.
    monkeypatch.setattr(stdio, "get_default_environment", lambda: {})
    params = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-c", server],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    with anyio.fail_after(20):
        async with stdio.stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                assert initialized.server_info.name == "synthetic-stdio"
                advertised = await session.list_tools()
                [schema] = advertised.tools
                assert schema.name == "fixture_echo"
                assert schema.input_schema["properties"]["value"]["type"] == "string"
                assert schema.input_schema["required"] == ["value"]
                direct = await session.call_tool("fixture_echo", {"value": "direct"})
                assert not direct.is_error
                assert any(
                    getattr(block, "text", "") == "fixture:direct" for block in direct.content
                )
                [adapted] = await load_mcp_tools(session)
                assert adapted.name == "fixture_echo"
                result = await adapted.ainvoke({"value": "adapted"})
                assert "fixture:adapted" in str(result)


def test_tool_graph_cost_callback_and_sqlite_resume(tmp_path: Path) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    from agents_gym_core.runtime.cost_tracking import CostTrackingCallback

    @tool
    def fixture_sum(left: int, right: int) -> int:
        """Add two synthetic fixture numbers."""
        return left + right

    usage = {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    model = FakeMessagesListChatModel(
        responses=[
            _call("fixture_sum", {"left": 2, "right": 3}, "sum", usage_metadata=usage),
            AIMessage(content="done", usage_metadata=usage),
        ]
    )
    graph = StateGraph(MessagesState)

    def model_node(state: MessagesState) -> MessagesState:
        return {"messages": [model.invoke(state["messages"])]}

    graph.add_node("model", model_node)
    graph.add_node("tools", ToolNode([fixture_sum]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    config: RunnableConfig = {"configurable": {"thread_id": "fixture-a"}}
    path = str(tmp_path / "checkpoint.sqlite")
    callback = CostTrackingCallback()
    callback_config: RunnableConfig = {**config, "callbacks": [callback]}
    with SqliteSaver.from_conn_string(path) as saver:
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke(
            {"messages": [HumanMessage(content="add")]},
            config=callback_config,
        )
        assert result["messages"][-1].content == "done"
        assert any(
            isinstance(message, ToolMessage) and message.content == "5"
            for message in result["messages"]
        )
    with SqliteSaver.from_conn_string(path) as saver:
        compiled = graph.compile(checkpointer=saver)
        assert compiled.get_state(config).values["messages"][-1].content == "done"
        assert not compiled.get_state({"configurable": {"thread_id": "fixture-b"}}).values
    assert callback.record.num_llm_calls == 2
    assert callback.record.input_tokens == 10
    assert callback.record.output_tokens == 4


class _ToolModel(FakeMessagesListChatModel):
    tool_snapshots: list[list[str]] = Field(default_factory=list)
    bind_settings: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _ToolModel:
        self.tool_snapshots.append([getattr(item, "name", "") for item in tools])
        self.bind_settings.append(kwargs)
        return self


class _Output(BaseModel):
    answer: str


@tool
def _fixture_lookup(query: str) -> str:
    """Look up a synthetic fixture value."""
    return "fixture"


def _scenario() -> Any:
    from uipath.eval.models.evaluation_set import EvaluationSet

    from agents_gym_core.utils.gym_types import Agent

    return Agent(
        name="fixture",
        tools=[_fixture_lookup],
        output_schema=_Output,
        input_schema=BaseModel,
        evaluation_set=EvaluationSet(id="fixture", name="fixture"),
        user_prompt="{query}",
        system_prompt="Synthetic fixture",
    )


def _call(name: str, args: dict[str, Any], call_id: str, **kwargs: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        **kwargs,
    )


_FILE_TOOLS = {"delete", "edit_file", "glob", "grep", "ls", "read_file", "write_file"}


@pytest.mark.parametrize("delegate_tools", [False, True])
@pytest.mark.asyncio
async def test_deep_agents_planning_and_tool_contract(delegate_tools: bool) -> None:
    from deepagents.backends import StateBackend

    from agents_gym_core.loops.deep_agents_loop import (
        create_deep_agents_graph,
        normalize_deep_agents_output,
    )

    model = _ToolModel(
        responses=[
            _call(
                "write_todos", {"todos": [{"content": "Fixture plan", "status": "completed"}]}, "p"
            ),
            _call("end_execution", {"answer": "done"}, "e"),
        ]
    )
    graph = create_deep_agents_graph(
        _scenario(), model, delegate_tools=delegate_tools, backend=StateBackend()
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="finish fixture")]})

    assert normalize_deep_agents_output(result) == {"answer": "done"}
    assert result["todos"] == [{"content": "Fixture plan", "status": "completed"}]
    expected = {*_FILE_TOOLS, "end_execution", "task", "write_todos"}
    if not delegate_tools:
        expected.update({"_fixture_lookup", "raise_error"})
    assert model.tool_snapshots
    assert all(set(names) == expected for names in model.tool_snapshots)
    assert all(len(names) == len(set(names)) for names in model.tool_snapshots)
    assert all(settings.get("parallel_tool_calls") is False for settings in model.bind_settings)


@pytest.mark.parametrize(
    ("delegate_tools", "subagent_type", "extra_tools"),
    [
        (False, "general-purpose", {"_fixture_lookup", "end_execution", "raise_error"}),
        (True, "general-purpose", {"end_execution"}),
        (True, "researcher", {"_fixture_lookup"}),
        (False, "fixture-specialist", {"_fixture_lookup"}),
    ],
)
@pytest.mark.asyncio
async def test_deep_agents_subagent_planning_through_task(
    delegate_tools: bool, subagent_type: str, extra_tools: set[str]
) -> None:
    from deepagents.backends import StateBackend

    from agents_gym_core.loops.deep_agents_loop import (
        create_deep_agents_graph,
        normalize_deep_agents_output,
    )

    model = _ToolModel(
        responses=[
            _call("task", {"description": "look up fixture", "subagent_type": subagent_type}, "t"),
            _call("write_todos", {"todos": [{"content": "Sub plan", "status": "completed"}]}, "p"),
            AIMessage(content="fixture result"),
            _call("end_execution", {"answer": "done"}, "e"),
        ]
    )
    specialist = {
        "name": "fixture-specialist",
        "description": "Synthetic specialist.",
        "system_prompt": "Synthetic specialist",
        "tools": [_fixture_lookup],
    }
    graph = create_deep_agents_graph(
        _scenario(),
        model,
        delegate_tools=delegate_tools,
        subagents=[specialist] if subagent_type == "fixture-specialist" else None,
        backend=StateBackend(),
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="finish fixture")]})

    assert normalize_deep_agents_output(result) == {"answer": "done"}
    task_reply = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "t"
    )
    assert "fixture result" in str(task_reply.content)
    snapshots = [names for names in model.tool_snapshots if "task" not in names]
    assert snapshots, f"{subagent_type} did not bind tools: {model.tool_snapshots}"
    assert all(set(names) == {*_FILE_TOOLS, "write_todos", *extra_tools} for names in snapshots)
    assert all(len(names) == len(set(names)) for names in snapshots)


def test_deep_agents_custom_backend_and_summary_settings() -> None:
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware

    from agents_gym_core.loops.deep_agents_loop import (
        _build_middleware,
        _translate_summarization_kwargs,
    )

    settings = {"max_tokens_before_summary": 500, "max_summary_tokens": 100}
    assert _translate_summarization_kwargs(settings) == {
        "trigger": ("tokens", 500),
        "trim_tokens_to_summarize": 100,
    }
    middleware = _build_middleware(
        FakeMessagesListChatModel(responses=[]), settings, StateBackend()
    )
    assert sum(isinstance(item, SummarizationMiddleware) for item in middleware) == 1


@pytest.mark.parametrize("parallel_tool_calls", [False, True])
@pytest.mark.asyncio
async def test_deep_agents_tool_control_preserves_other_model_settings(
    parallel_tool_calls: bool,
) -> None:
    from langchain.agents.middleware.types import ModelRequest

    from agents_gym_core.loops.deep_agents_loop import _GymToolControlMiddleware

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[]),
        messages=[HumanMessage(content="synthetic request")],
        model_settings={"temperature": 0.6, "parallel_tool_calls": True},
    )
    handler = AsyncMock()
    middleware = _GymToolControlMiddleware(parallel_tool_calls=parallel_tool_calls)
    await middleware.awrap_model_call(request, handler)
    assert handler.await_args is not None
    forwarded = handler.await_args.args[0]
    assert forwarded.model_settings == {
        "temperature": 0.6,
        "parallel_tool_calls": parallel_tool_calls,
    }
    assert request.model_settings == {"temperature": 0.6, "parallel_tool_calls": True}


def test_deep_agents_subagent_planning_is_once_and_preserves_caller_specs() -> None:
    from deepagents.backends import StateBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import _build_middleware, _with_planning_subagents

    model = FakeMessagesListChatModel(responses=[])
    middleware = _build_middleware(
        model,
        {"max_tokens_before_summary": 500, "max_summary_tokens": 100},
        StateBackend(),
    )
    assert sum(isinstance(item, TodoListMiddleware) for item in middleware) == 1
    summary = next(item for item in middleware if isinstance(item, SummarizationMiddleware))
    [general] = _with_planning_subagents([], middleware, ["/fixture-skills/"], model)
    assert general["name"] == "general-purpose"
    assert "tools" not in general
    assert general["skills"] == ["/fixture-skills/"]
    assert any(item is summary for item in general["middleware"])
    assert sum(isinstance(item, TodoListMiddleware) for item in general["middleware"]) == 1

    todos = TodoListMiddleware()
    custom = {"name": "general-purpose", "middleware": [todos]}
    [preserved] = _with_planning_subagents([custom], middleware, None, model)
    assert preserved is custom
    assert preserved["middleware"] == [todos]
    untouched = {"name": "specialist", "description": "fixture", "system_prompt": "fixture"}
    _, planned = _with_planning_subagents([untouched], middleware, None, model)
    assert "middleware" not in untouched
    assert len(planned["middleware"]) == 1
    for caller_owned in [
        {"name": "compiled", "runnable": object()},
        {"name": "remote", "graph_id": "synthetic-not-invoked"},
    ]:
        _, unchanged = _with_planning_subagents([caller_owned], middleware, None, model)
        assert unchanged is caller_owned


class _ProfileModel(_ToolModel):
    model_name: str = "main"
    provider: str
    descriptions: list[dict[str, str]] = Field(default_factory=list)
    system_prompts: list[Any] = Field(default_factory=list)

    def _get_ls_params(self, stop: Any = None, **kwargs: Any) -> Any:
        return {"ls_provider": self.provider, "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _ProfileModel:
        self.descriptions.append({item.name: item.description for item in tools})
        super().bind_tools(tools, **kwargs)
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        self.system_prompts.append(messages[0].content)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def profile_provider() -> str:
    # Public registrations are unique so these fixtures cannot change another test's profile.
    return f"sdk-contract-{uuid4().hex}"


@pytest.mark.parametrize("delegate_tools", [False, True])
@pytest.mark.asyncio
async def test_profile_can_disable_general_purpose_without_disabling_explicit_agents(
    profile_provider: str, delegate_tools: bool
) -> None:
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    register_harness_profile(
        profile_provider,
        HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
    )
    responses: list[BaseMessage] = [AIMessage(content="done")]
    if delegate_tools:
        responses = [
            _call("task", {"description": "fixture", "subagent_type": "general-purpose"}, "t"),
            _call("end_execution", {"answer": "done"}, "e"),
        ]
    model = _ProfileModel(provider=profile_provider, responses=responses)
    graph = create_deep_agents_graph(_scenario(), model, delegate_tools=delegate_tools)
    result = await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    tools = model.descriptions[0]
    if delegate_tools:
        assert "researcher" in tools["task"]
        reply = next(
            m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "t"
        )
        assert "general-purpose because it does not exist" in str(reply.content)
    else:
        assert "task" not in tools
    assert "write_todos" in tools


@pytest.mark.parametrize("model_override", [False, True])
@pytest.mark.asyncio
async def test_profile_gp_description_and_model_specific_enabled_override(
    profile_provider: str, model_override: bool
) -> None:
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    register_harness_profile(
        profile_provider,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                enabled=not model_override, description="provider GP description"
            )
        ),
    )
    if model_override:
        register_harness_profile(
            f"{profile_provider}:main",
            HarnessProfile(
                general_purpose_subagent=GeneralPurposeSubagentProfile(
                    enabled=True, description="model GP description"
                )
            ),
        )
    model = _ProfileModel(provider=profile_provider, responses=[AIMessage(content="done")])
    graph = create_deep_agents_graph(_scenario(), model)
    await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    expected = "model GP description" if model_override else "provider GP description"
    assert expected in model.descriptions[0]["task"]
    assert "write_todos" in model.descriptions[0]


@pytest.mark.parametrize("base_prompt", [None, "provider base prompt"])
@pytest.mark.parametrize("profile_planning", [False, True])
@pytest.mark.asyncio
async def test_gp_specific_prompt_uses_native_assembly_and_planning_opt_out(
    profile_provider: str, base_prompt: str | None, profile_planning: bool
) -> None:
    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        create_deep_agent,
        register_harness_profile,
    )
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    profile_todos: list[Any] = (
        [TodoListMiddleware(system_prompt="custom todo prompt")] if profile_planning else []
    )
    register_harness_profile(
        profile_provider,
        HarnessProfile(
            base_system_prompt=base_prompt,
            system_prompt_suffix="profile suffix",
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                description="custom GP description", system_prompt="GP-specific prompt"
            ),
            extra_middleware=profile_todos,
        ),
    )
    observed = []
    for core in [False, True]:
        model = _ProfileModel(
            provider=profile_provider,
            responses=[
                _call("task", {"description": "fixture", "subagent_type": "general-purpose"}, "t"),
                AIMessage(content="sub-agent result"),
                _call("end_execution", {"answer": "done"}, "e")
                if core
                else AIMessage(content="done"),
            ],
        )
        graph = (
            create_deep_agents_graph(_scenario(), model) if core else create_deep_agent(model=model)
        )
        await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
        assert "custom GP description" in model.descriptions[0]["task"]
        assert ("write_todos" in model.descriptions[1]) is profile_planning
        observed.append(model.system_prompts[1])
    assert observed[0] == observed[1]
    assert "GP-specific prompt" in str(observed[1])
    assert "provider base prompt" not in str(observed[1])
    assert str(observed[1]).count("profile suffix") == 1


@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize(
    ("delegate_tools", "subagent_type"),
    [
        (False, "general-purpose"),
        (True, "general-purpose"),
        (True, "researcher"),
        (False, "specialist"),
    ],
)
@pytest.mark.asyncio
async def test_profile_todo_customization_survives_on_coordinator_and_subagents(
    profile_provider: str, factory: bool, delegate_tools: bool, subagent_type: str
) -> None:
    from deepagents import HarnessProfile, register_harness_profile
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    calls = []

    def configured_todos() -> list[Any]:
        calls.append(True)
        return [
            TodoListMiddleware(system_prompt="profile TODO prompt", tool_description="profile TODO")
        ]

    register_harness_profile(
        profile_provider,
        HarnessProfile(extra_middleware=configured_todos if factory else configured_todos()),
    )
    calls.clear()
    model = _ProfileModel(
        provider=profile_provider,
        responses=[
            _call("task", {"description": "fixture", "subagent_type": subagent_type}, "t"),
            _call("write_todos", {"todos": [{"content": "fixture", "status": "completed"}]}, "p"),
            AIMessage(content="sub-agent result"),
            _call("end_execution", {"answer": "done"}, "e"),
        ],
    )
    specialist = {"name": "specialist", "description": "fixture", "system_prompt": "fixture"}
    graph = create_deep_agents_graph(
        _scenario(),
        model,
        delegate_tools=delegate_tools,
        subagents=[specialist] if subagent_type == "specialist" else None,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "t" for m in result["messages"])
    assert all(tools["write_todos"] == "profile TODO" for tools in model.descriptions)
    assert all("profile TODO prompt" in str(prompt) for prompt in model.system_prompts)
    assert all(names.count("write_todos") == 1 for names in model.tool_snapshots)
    if factory:
        assert len(calls) == (3 if delegate_tools or subagent_type == "specialist" else 2)
    else:
        assert not calls


@pytest.mark.parametrize("caller_override", [False, True])
@pytest.mark.asyncio
async def test_model_specific_and_explicit_caller_planning_precedence(
    profile_provider: str, caller_override: bool
) -> None:
    from deepagents import HarnessProfile, register_harness_profile
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    provider_todos: list[Any] = [TodoListMiddleware(tool_description="provider")]
    register_harness_profile(profile_provider, HarnessProfile(extra_middleware=provider_todos))
    for name in ["main", "child"]:
        model_todos: list[Any] = [TodoListMiddleware(tool_description=name)]
        register_harness_profile(
            f"{profile_provider}:{name}",
            HarnessProfile(extra_middleware=model_todos),
        )
    parent = _ProfileModel(
        provider=profile_provider,
        responses=[
            _call("task", {"description": "fixture", "subagent_type": "specialist"}, "t"),
            _call("end_execution", {"answer": "done"}, "e"),
        ],
    )
    child = _ProfileModel(
        provider=profile_provider, model_name="child", responses=[AIMessage(content="result")]
    )
    child_middleware = (
        [TodoListMiddleware(tool_description="caller child")] if caller_override else []
    )
    spec = {
        "name": "specialist",
        "description": "fixture",
        "system_prompt": "fixture",
        "model": child,
        "middleware": child_middleware,
    }
    caller = [TodoListMiddleware(tool_description="caller main")] if caller_override else []
    graph = create_deep_agents_graph(_scenario(), parent, subagents=[spec], middleware=caller)
    await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    assert all(
        tools["write_todos"] == ("caller main" if caller_override else "main")
        for tools in parent.descriptions
    )
    assert child.descriptions[0]["write_todos"] == ("caller child" if caller_override else "child")
    assert spec["middleware"] is child_middleware


def test_profile_planning_helpers_do_not_construct_string_models_or_touch_owned_graphs(
    profile_provider: str,
) -> None:
    from deepagents import HarnessProfile, register_harness_profile
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import _with_planning

    string_todos: list[Any] = [TodoListMiddleware(tool_description="string profile")]
    register_harness_profile(
        f"{profile_provider}:unconstructed",
        HarnessProfile(extra_middleware=string_todos),
    )
    model = _ProfileModel(provider=profile_provider, responses=[])
    spec = {"name": "string-model", "model": f"{profile_provider}:unconstructed"}
    assert _with_planning(spec, model) is spec
    for owned in [
        {"name": "compiled", "runnable": object()},
        {"name": "remote", "graph_id": "not-invoked"},
    ]:
        assert _with_planning(owned, model) is owned


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.asyncio
async def test_explicit_gp_overrides_disabled_default_and_preserves_owned_graph(
    profile_provider: str, compiled: bool
) -> None:
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    profile_todos: list[Any] = [TodoListMiddleware(tool_description="profile TODO")]
    register_harness_profile(
        profile_provider,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                enabled=False,
                description="profile GP description",
                system_prompt="profile GP prompt",
            ),
            extra_middleware=profile_todos,
        ),
    )
    parent = _ProfileModel(
        provider=profile_provider,
        responses=[
            _call("task", {"description": "fixture", "subagent_type": "general-purpose"}, "t"),
            _call("end_execution", {"answer": "done"}, "e"),
        ],
    )
    child = _ProfileModel(provider=profile_provider, responses=[AIMessage(content="result")])
    spec: dict[str, Any] = {"name": "general-purpose", "description": "caller GP description"}
    if compiled:
        spec["runnable"] = create_agent(child, system_prompt="caller GP prompt")
    else:
        spec.update(
            model=child,
            system_prompt="caller GP prompt",
            middleware=[TodoListMiddleware(tool_description="caller TODO")],
        )
    graph = create_deep_agents_graph(_scenario(), parent, subagents=[spec])
    await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    assert "caller GP description" in parent.descriptions[0]["task"]
    assert "profile GP description" not in parent.descriptions[0]["task"]
    assert "caller GP prompt" in str(child.system_prompts[0])
    assert "profile GP prompt" not in str(child.system_prompts[0])
    if compiled:
        assert not child.descriptions
    else:
        assert child.descriptions[0]["write_todos"] == "caller TODO"


@pytest.mark.asyncio
async def test_callable_profile_without_todos_keeps_native_planning_opt_out(
    profile_provider: str,
) -> None:
    from deepagents import HarnessProfile, register_harness_profile

    from agents_gym_core.loops.deep_agents_loop import create_deep_agents_graph

    calls = []

    def native_factory() -> list[Any]:
        calls.append(True)
        return []

    register_harness_profile(profile_provider, HarnessProfile(extra_middleware=native_factory))
    model = _ProfileModel(provider=profile_provider, responses=[AIMessage(content="done")])
    graph = create_deep_agents_graph(_scenario(), model)
    await graph.ainvoke({"messages": [HumanMessage(content="synthetic fixture")]})
    assert "write_todos" not in model.descriptions[0]
    assert len(calls) == 2  # Native GP and coordinator stacks only.


@pytest.mark.parametrize("delegate_tools", [False, True])
@pytest.mark.asyncio
async def test_deep_agents_delete_is_confined_to_configured_state_backend(
    delegate_tools: bool, tmp_path: Path
) -> None:
    from deepagents.backends import StateBackend

    from agents_gym_core.loops.deep_agents_loop import (
        create_deep_agents_graph,
        normalize_deep_agents_output,
    )

    # The identical path exists in graph state and on disk. Only state may change.
    host_file = tmp_path / "synthetic.txt"
    host_file.write_text("host fixture remains", encoding="utf-8")
    path = host_file.as_posix()
    model = _ToolModel(
        responses=[
            _call("write_file", {"file_path": path, "content": "state fixture"}, "w"),
            _call("delete", {"file_path": path}, "d"),
            _call("end_execution", {"answer": "done"}, "e"),
        ]
    )
    graph = create_deep_agents_graph(
        _scenario(), model, delegate_tools=delegate_tools, backend=StateBackend()
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="delete state fixture")]})
    assert normalize_deep_agents_output(result) == {"answer": "done"}
    deleted = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "d"
    )
    assert deleted.status == "success"
    assert path not in result.get("files", {})
    assert host_file.read_text(encoding="utf-8") == "host fixture remains"
