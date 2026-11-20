"""Deep Agents loop — wraps LangChain AI's `create_deep_agent` harness.

`create_deep_agent` (from the `deepagents` library) is a batteries-included
agent harness built on top of LangGraph. This Core integration provides:

- **Planning** — Core adds `write_todos` defaults where the resolved profile
  permits; custom GP prompts and callable middleware own their planning policy.
- **Filesystem** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`
  for reading/writing context to an in-memory (or pluggable) backend.
- **Sub-agents** — a `task` tool that delegates work to isolated sub-agents,
  preventing context bloat on long tasks.
- **Auto-summarization** — long conversations are automatically summarised to
  keep the context window healthy.

## Gym integration

UiPath flow control tools (``end_execution``, ``raise_error``) are injected
into the agent's tool list, generated from the output schema so
``end_execution`` accepts exactly the right fields.  Both are marked
``return_direct=True`` so the graph exits immediately when called — this
uses ``create_agent``'s own exit condition #2 in ``_make_tools_to_model_edge``.

A system-prompt instruction directs the model to submit its final output
through ``end_execution`` using the agent's output fields.

``normalize_deep_agents_output`` finds the ``end_execution`` ``ToolMessage``
and returns its JSON content directly.

Parallel tool calls are disabled by default via ``_GymToolControlMiddleware``
(matching the gym's ``UIPATH_AGENTS_GYM_PARALLEL_TOOL_CALLS`` setting).

Usage in ``agent.yaml``::

    loop: deep-agents
"""

import json
from collections.abc import Callable
from typing import Any

from deepagents import HarnessProfile, SubAgent, create_deep_agent
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.profiles.harness.harness_profiles import (
    _get_harness_profile,
    _harness_profile_for_model,
)
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph, StateGraph
from uipath.runtime.errors.contract import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.react.tools import create_flow_control_tools
from uipath_langchain.agent.react.types import AgentGraphConfig

from agents_gym_core.loops.loop_directory_provider import LoopDirectoryProvider
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION, LoopDetails, LoopName
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.prompt_loader import PromptLoader

_END_EXECUTION_INSTRUCTION = (
    "\n\nYou MUST call end_execution for output — never plain text. IGNORE conversational elements."
)

# Dedicated summary prompt for deepagents' SummarizationMiddleware.
# Unlike the langmem SummarizationNode (which prepends messages as actual chat
# messages and puts the instruction text last), SummarizationMiddleware formats
# everything into a single string via prompt.format(messages=...).  The prompt
# must therefore contain the {messages} placeholder exactly once.
_DEEP_AGENTS_SUMMARY_PROMPT = PromptLoader.load_prompt_from_file(
    "core::summary_deepagents_prompt.md", "node_prompts", LoopDirectoryProvider()
)

_GYM_ONLY_KWARGS = frozenset(
    {
        "planning_llm",
        "validation_llm",
        "reasoning_llm",
        "summarizing_llm",
        "observer_llm",
        "checker_llm",
        "embeddings",
        "config",
        "checkpointer",
        "print_trace",
        "tool_choice",
        "strict",
        "parallel_tool_calls",
        "guardrails",
        "handle_tool_errors",
        "graceful_termination",
        "detect_tool_repetition",
        "with_memory",
        "memory_tools",
        "planner_system_prompt",
        "planner_user_prompt",
        "plan_insertion_prompt",
        "validator_system_prompt",
        "validator_user_prompt",
        "inject_agent_instructions",
        "reasoning_prompt",
        "initial_summary_prompt",
        "existing_summary_prompt",
        "max_tokens_before_summary",
        "max_tokens",
        "summarization_model",
        "max_summary_tokens",
        "token_counter",
        "query_limit",
        "middleware",
    }
)


def _build_tools(scenario: Agent) -> list[BaseTool]:
    """Combine the agent's tools with UiPath flow control tools (``return_direct=True``)."""
    flow_control = create_flow_control_tools(scenario.output_schema)
    for tool in flow_control:
        tool.return_direct = True
    return [*scenario.tools, *flow_control]


def _translate_summarization_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate gym summarization settings to deepagents ``SummarizationMiddleware`` kwargs."""
    translated: dict[str, Any] = {}
    if "max_tokens_before_summary" in kwargs:
        translated["trigger"] = ("tokens", kwargs["max_tokens_before_summary"])
    if "max_summary_tokens" in kwargs:
        translated["trim_tokens_to_summarize"] = kwargs["max_summary_tokens"]
    if "token_counter" in kwargs:
        translated["token_counter"] = kwargs["token_counter"]
    return translated


def _filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Strip gym-internal kwargs not accepted by ``create_deep_agent``."""
    return {k: v for k, v in kwargs.items() if k not in _GYM_ONLY_KWARGS}


def _make_loop_details(model: BaseChatModel) -> LoopDetails:
    return LoopDetails(
        loop_name=LoopName.DEEP_AGENTS,
        loop_version=DEFAULT_LOOP_VERSION,
        models={"model": get_model_name(model)},
    )


class _GymToolControlMiddleware(AgentMiddleware):
    """Disables parallel tool calls via ``model_settings``."""

    def __init__(self, *, parallel_tool_calls: bool = True):
        self._parallel_tool_calls = parallel_tool_calls

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ) -> ModelResponse:
        if not self._parallel_tool_calls:
            request = request.override(
                model_settings={**(request.model_settings or {}), "parallel_tool_calls": False}
            )
        return await handler(request)


def _build_middleware(
    model: BaseChatModel,
    kwargs: dict[str, Any],
    backend: BackendProtocol | None,
    *,
    parallel_tool_calls: bool = True,
) -> list[Any]:
    """Build the middleware list for the deep-agents graph."""
    caller_middleware = list(kwargs.get("middleware") or [])
    middleware: list[Any] = [
        _GymToolControlMiddleware(parallel_tool_calls=parallel_tool_calls),
        *_planning_defaults(_profile_for_model(model), caller_middleware),
        *caller_middleware,
    ]
    summ_kwargs = _translate_summarization_kwargs(kwargs)
    if summ_kwargs and backend is not None:
        middleware.append(
            SummarizationMiddleware(
                model=model,
                backend=backend,
                summary_prompt=_DEEP_AGENTS_SUMMARY_PROMPT,
                **summ_kwargs,
            )
        )
    return middleware


def _profile_for_model(model: BaseChatModel | str) -> HarnessProfile:
    """Use the retained SDK's exact provider/model profile resolution rules."""
    if isinstance(model, str):
        # The SDK resolves a string model against its original spec, not its
        # constructed identifier. Looking up that spec needs no model construction.
        return _get_harness_profile(model) or HarnessProfile()
    return _harness_profile_for_model(model, None)


def _planning_defaults(profile: HarnessProfile, middleware: list[Any]) -> list[Any]:
    """Do not override planning owned by a caller or the resolved model profile."""
    if any(isinstance(m, TodoListMiddleware) for m in middleware):
        return []
    # Factories may have side effects. Only the SDK may materialize them; a
    # callable profile owns planning rather than receiving a Core override.
    if callable(profile.extra_middleware):
        return []
    if any(isinstance(m, TodoListMiddleware) for m in profile.extra_middleware):
        return []
    return [TodoListMiddleware()]


def _with_planning(spec: Any, model: BaseChatModel) -> Any:
    """Keep planning on declarative sub-agents without changing caller-owned graphs."""
    if "runnable" in spec or "graph_id" in spec:
        return spec
    existing = list(spec.get("middleware", []))
    planning = _planning_defaults(_profile_for_model(spec.get("model", model)), existing)
    if not planning:
        return spec
    return {**spec, "middleware": [*planning, *existing]}


def _with_planning_subagents(
    subagents: list[Any],
    extra_middleware: list[Any],
    skills: list[str] | None,
    model: BaseChatModel,
) -> list[Any]:
    """Add planning to sub-agents, including the default general-purpose sub-agent.

    An explicit default spec adds planning while retaining tools, skills and
    summarization inheritance. Profiles with a GP-specific prompt or callable
    middleware keep native assembly and own their planning configuration.
    """
    planned = [_with_planning(spec, model) for spec in subagents]
    if not any(spec.get("name") == GENERAL_PURPOSE_SUBAGENT["name"] for spec in planned):
        profile = _profile_for_model(model)
        gp_profile = profile.general_purpose_subagent
        if gp_profile is not None and gp_profile.enabled is False:
            return planned
        if gp_profile is not None and gp_profile.system_prompt is not None:
            # Native GP prompt precedence differs from an explicit declarative
            # spec. Preserve it instead of rewriting model-request messages.
            return planned
        planning = _planning_defaults(profile, [])
        if not planning:
            # Native assembly owns planning and preserves the GP policy.
            return planned
        inherited = [m for m in extra_middleware if isinstance(m, SummarizationMiddleware)]
        general_purpose: SubAgent = {
            **GENERAL_PURPOSE_SUBAGENT,
            "middleware": [*planning, *inherited],
        }
        if gp_profile is not None:
            if gp_profile.description is not None:
                general_purpose["description"] = gp_profile.description
        if skills is not None:
            general_purpose["skills"] = skills
        planned.insert(0, general_purpose)
    return planned


_DEFAULT_RESEARCHER_PROMPT: str = """\
You are a document research specialist. Your job is to find specific \
information in a document corpus using the available search and retrieval tools.

WORKFLOW:
1. Read the task carefully. Note EXACTLY what is being asked for.
2. Search using the available search tools with specific keywords from the task.
3. If a document loading tool is available, load the full content of the \
   most relevant pages (±2 pages around each hit for context).
4. Search AGAIN with different terms (synonyms, related concepts, \
   specific names/numbers from the first results).
5. Before reporting, re-read the original question and check: \
   - Have I found the SPECIFIC fact/number/list that was asked for? \
   - If a list was requested, have I found ALL items, not just the first few? \
   - Am I quoting from the right section, not an adjacent table/chart?
6. Report the relevant facts, quoting exact text with source citations.

RULES:
- Run at least 2 searches with different keywords.
- Quote ONLY information from the documents. NEVER add your own knowledge.
- Include exact numbers, names, dates, percentages as they appear in the text.
- Include source citations for every quoted fact.
- If a page has multiple similar facts/numbers, quote ONLY the one that \
  matches what the question asks for.
- If you cannot find the information, say so clearly.
"""


def create_deep_agents_graph(
    scenario: Agent,
    model: BaseChatModel,
    *,
    config: AgentGraphConfig | None = None,
    subagents: list[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    delegate_tools: bool = False,
    researcher_prompt: str | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build a compiled Deep Agents graph for the given scenario.

    Args:
        delegate_tools: When ``True``, the coordinator receives NO scenario
            tools, but retains built-in planning/filesystem tools,
            ``end_execution`` and the ``task`` tool. A
            "researcher" sub-agent is created with the scenario's tools.
            This forces the coordinator to delegate all research via
            sub-agents, preventing single-search-and-answer patterns.
        researcher_prompt: Custom system prompt for the researcher sub-agent
            (only used when ``delegate_tools=True``).
    """
    parallel_tool_calls = config.parallel_tool_calls if config is not None else False
    backend = kwargs.get("backend")
    extra_middleware = _build_middleware(
        model,
        kwargs,
        backend,
        parallel_tool_calls=parallel_tool_calls,
    )

    if delegate_tools:
        # Forced delegation: coordinator gets only flow control,
        # researcher sub-agent gets the scenario's tools.
        flow_control = [
            t for t in create_flow_control_tools(scenario.output_schema) if t.name != "raise_error"
        ]
        for t in flow_control:
            t.return_direct = True

        tool_names = [t.name for t in scenario.tools]
        tools_desc = ", ".join(tool_names) if tool_names else "the scenario's tools"

        researcher: SubAgent = {
            "name": "researcher",
            "description": (
                f"Document research specialist. Delegate ALL search and document "
                f"retrieval tasks to this agent. It has access to {tools_desc} "
                f"for thorough corpus research."
            ),
            "system_prompt": researcher_prompt or _DEFAULT_RESEARCHER_PROMPT,
            "tools": list(scenario.tools),
        }

        # Prepend researcher to any user-provided subagents
        all_subagents: list[Any] = [researcher]
        if subagents:
            all_subagents.extend(subagents)

        coordinator_tools = flow_control
        effective_subagents = all_subagents
    else:
        coordinator_tools = _build_tools(scenario)
        effective_subagents = list(subagents or [])

    effective_subagents = _with_planning_subagents(
        effective_subagents, extra_middleware, skills, model
    )

    compiled = create_deep_agent(
        model=model,
        tools=coordinator_tools,
        system_prompt=scenario.system_prompt + _END_EXECUTION_INSTRUCTION,
        subagents=effective_subagents,
        skills=skills,
        memory=memory,
        middleware=extra_middleware,
        checkpointer=kwargs.get("checkpointer"),
        **_filter_kwargs(kwargs),
    )
    compiled.loop_details = _make_loop_details(model)  # pyright: ignore[reportAttributeAccessIssue]
    return compiled


def create_deep_agents_uncompiled_graph(
    scenario: Agent,
    model: BaseChatModel,
    *,
    config: AgentGraphConfig | None = None,
    subagents: list[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    delegate_tools: bool = False,
    researcher_prompt: str | None = None,
    **kwargs: Any,
) -> StateGraph:
    """Build an uncompiled Deep Agents graph via :func:`create_deep_agents_graph`."""
    compiled = create_deep_agents_graph(
        scenario,
        model,
        config=config,
        subagents=subagents,
        skills=skills,
        memory=memory,
        delegate_tools=delegate_tools,
        researcher_prompt=researcher_prompt,
        **kwargs,
    )
    builder: StateGraph = compiled.builder  # pyright: ignore[reportAttributeAccessIssue]
    builder.loop_details = compiled.loop_details  # pyright: ignore[reportAttributeAccessIssue]
    return builder


def _tool_call_args(messages: list[Any], tool_call_id: str) -> dict[str, Any]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                if call.get("id") == tool_call_id:
                    return call.get("args") or {}
    return {}


def normalize_deep_agents_output(result: dict[str, Any]) -> dict[str, Any]:
    """Extract structured output from a deep-agents ``ainvoke`` result.

    Primary: ``end_execution`` ToolMessage. Fallback: last AIMessage text. Both flow control
    tools return directly, so an ``end_execution`` that failed the output schema and a
    ``raise_error`` also end the run; they become the error outputs the other loops produce
    (``OUTPUT_VALIDATION_ERROR``, ``TERMINATION_LLM_RAISED_ERROR``).
    """
    messages = result.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "raise_error":
            args = _tool_call_args(messages, msg.tool_call_id)
            return {
                "type": "error",
                "code": AgentRuntimeError.full_code(
                    AgentRuntimeErrorCode.TERMINATION_LLM_RAISED_ERROR
                ),
                "title": args.get("message", "The LLM did not set the error message"),
                "category": UiPathErrorCategory.USER,
                "details": args.get("details", ""),
            }
        if isinstance(msg, ToolMessage) and msg.name == "end_execution" and msg.status == "error":
            return {
                "type": "error",
                "code": AgentRuntimeError.full_code(AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR),
                "title": "Agent output did not match the expected schema",
                "category": UiPathErrorCategory.USER,
                "details": str(msg.content),
            }
        if isinstance(msg, ToolMessage) and msg.name == "end_execution":
            content = msg.content
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"content": content}
            return {"content": str(content)}

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            stripped = content.strip()
            if stripped.startswith("{"):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            return {"content": content}

    return {}


def build_deep_agents_input(scenario: Agent, input_data: dict[str, Any]) -> dict[str, Any]:
    """Convert gym input dict to ``{"messages": [HumanMessage(...)]}``."""
    try:
        user_content = scenario.user_prompt.format_map(input_data)
    except KeyError:
        user_content = json.dumps(input_data)

    return {"messages": [HumanMessage(content=user_content)]}


__all__ = [
    "build_deep_agents_input",
    "create_deep_agents_graph",
    "create_deep_agents_uncompiled_graph",
    "normalize_deep_agents_output",
]
