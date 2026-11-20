"""Wrap an agent graph as a ``BaseTool`` so another agent can call it as a subagent.

The subagent runs its configured graph (a ReAct loop, a coded workflow or a graph
factory) and returns its output to the parent. All parameters can be set from
``agent.yaml``::

    tools:
      - _target_: agents_gym_core.common_tools.agent_as_tool.AgentAsTool
        name: "my_subagent"
        description: "Delegates deep investigation to a specialist subagent."
        model_name: "gpt-5-mini-2025-08-07"
        scenario: ...
        # loop_class, loop_kwargs, recursion_limit all optional
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool
from pydantic import Field, model_validator

from agents_gym_core.loops.basic_loop import BasicLoop, BasicLoopAny
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.utils.gym_settings_core import gym_settings_core
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import warning

DEFAULT_SUBAGENT_MODEL = "gpt-5-mini-2025-08-07"


class AgentAsToolError(RuntimeError):
    """Raised instead of returning an error payload when ``raise_on_error`` is set."""


class AgentAsTool(BaseTool):
    """A ``BaseTool`` that runs a subagent graph to completion.

    The subagent has its own ``Agent`` scenario (prompts, tools, schemas). The parent
    sees one tool call; the tool arguments default to ``scenario.input_schema``.

    Args:
        model_name: Resolved with ``get_chat_model()`` unless ``model`` is given.
        loop_class: Used when the scenario configures no ``loop``.
        loop_kwargs: Extra keyword arguments for the loop.
        recursion_limit: Maximum LangGraph steps.
        forward_configurable: Pass the parent's ``configurable`` to the subagent.
        result_field: Return only this field of the subagent's output.
        raise_on_error: Raise ``AgentAsToolError`` instead of returning ``{"error": ...}``.
    """

    name: str = "subagent"
    description: str = "Delegates work to a specialist subagent."

    scenario: Agent = Field(exclude=True)
    model_name: str = DEFAULT_SUBAGENT_MODEL
    loop_class: type[BasicLoop] = Field(default=BasicLoopAny, exclude=True)
    loop_kwargs: dict[str, Any] = Field(default_factory=dict)
    recursion_limit: int = gym_settings_core.UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT
    model: BaseChatModel | None = Field(default=None, exclude=True)
    forward_configurable: bool = Field(default=False, exclude=True)
    result_field: str = Field(default="", exclude=True)
    raise_on_error: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _infer_args_schema(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Default ``args_schema`` to ``scenario.input_schema`` when not provided."""
        if "args_schema" not in values and "scenario" in values:
            scenario = values["scenario"]
            if hasattr(scenario, "input_schema"):
                values["args_schema"] = scenario.input_schema
        return values

    def _run(self, config: RunnableConfig | None = None, **kwargs: Any) -> str:
        """Run ``_arun`` in a new event loop."""
        return asyncio.run(self._arun(config=config, **kwargs))

    async def _arun(self, config: RunnableConfig | None = None, **kwargs: Any) -> str:
        """Run the subagent and return its output as JSON (or ``result_field`` alone)."""
        model = self.model or get_chat_model(self.model_name)
        if self.scenario.loop:
            from agents_gym_core.loops import get_loop

            loop_name, separator, configured_version = self.scenario.loop.partition(":")
            graph = get_loop(
                loop_name=loop_name,
                version=configured_version if separator else DEFAULT_LOOP_VERSION,
                scenario=self.scenario,
                models={"model": model},
                **self.loop_kwargs,
            )
        else:
            loop = self.loop_class(scenario=self.scenario, model=model, **self.loop_kwargs)
            graph = loop.build_compiled_graph()

        # Strip injected keys the subagent's input schema doesn't expect
        input_data = {k: v for k, v in kwargs.items() if k != "config"}

        configurable: dict[str, Any] = {"thread_id": str(uuid4())}
        parent_config = config or var_child_runnable_config.get(None)
        if self.forward_configurable and parent_config is not None:
            parent_configurable = parent_config.get("configurable")
            if isinstance(parent_configurable, dict):
                configurable.update(parent_configurable)
            configurable["thread_id"] = str(configurable.get("thread_id") or uuid4())

        try:
            result = await graph.ainvoke(
                input_data,
                config={
                    "configurable": configurable,
                    "recursion_limit": self.recursion_limit,
                },
            )
        except Exception as exc:
            message = f"Subagent failed: {exc}"
            warning(f"Subagent '{self.name}' failed: {exc}")
            if self.raise_on_error:
                raise AgentAsToolError(message) from exc
            return json.dumps({"error": message})

        if result is None:
            message = "Subagent returned no result"
            if self.raise_on_error:
                raise AgentAsToolError(message)
            return json.dumps({"error": message})

        if isinstance(result, dict):
            if self.result_field:
                if self.result_field not in result:
                    message = f"Subagent result has no field {self.result_field!r}"
                    if self.raise_on_error:
                        raise AgentAsToolError(message)
                    return json.dumps({"error": message})
                value = result[self.result_field]
                return value if isinstance(value, str) else json.dumps(value, default=str)
            # Drop internal state keys (leading underscore).
            output = {k: v for k, v in result.items() if not k.startswith("_")}
            return json.dumps(output, default=str)

        return str(result)
