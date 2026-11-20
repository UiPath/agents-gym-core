"""Wrap an existing Python agent in the Gym runner and prompt optimizer."""

import asyncio
import hashlib
import inspect
import sys
from collections.abc import Callable
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, create_model

from agents_gym_core.utils.gym_types import Agent


class CallableAgent(Agent):
    entrypoint: str


def resolve_entrypoint(path: str) -> Callable:
    if ":" in path:
        filename, _, name = path.rpartition(":")
        file = Path(filename)
        module_name = "_gym_adapter_" + hashlib.sha256(str(file).encode()).hexdigest()
        if module_name not in sys.modules:
            spec = spec_from_file_location(module_name, file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import adapter: {filename}")
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        func = getattr(sys.modules[module_name], name)
    else:
        module_name, sep, name = path.rpartition(".")
        if not sep:
            raise ValueError("agent.entrypoint must be a dotted callable or Python-file:function")
        func = getattr(import_module(module_name), name)
    if not callable(func) or inspect.isclass(func):
        raise TypeError(f"{path} must be a function or callable instance")
    # Fail during setup, before scoring or spending tokens, on an incompatible adapter.
    inspect.signature(func).bind(inputs={}, system_prompt="", user_prompt="", model=None)
    return func


def create_callable_graph(scenario: CallableAgent, model: BaseChatModel) -> StateGraph:
    """Call ``entrypoint(*, inputs, system_prompt, user_prompt, model) -> dict``.

    Both sync and async adapters are supported. The prompt arguments always contain
    the selected version's current text, including the optimizer's edits. Returned
    data must match the agent's output schema; adapter exceptions remain run errors.
    """
    func = resolve_entrypoint(scenario.entrypoint)
    overlap = scenario.input_schema.model_fields.keys() & scenario.output_schema.model_fields.keys()
    if overlap:
        raise ValueError(
            f"Callable agent input and output fields must be distinct: {sorted(overlap)}"
        )
    # Keep the original input model (constraints, defaults and validators included).
    fields: dict[str, Any] = {
        name: (field.annotation or Any, None)
        for name, field in scenario.output_schema.model_fields.items()
    }
    state_schema = create_model("CallableState", __base__=scenario.input_schema, **fields)

    async def invoke(state: BaseModel) -> dict:
        raw_inputs = {
            key: value
            for key, value in state.model_dump().items()
            if key in scenario.input_schema.model_fields
        }
        inputs = scenario.input_schema.model_validate(raw_inputs).model_dump()
        kwargs = dict(
            inputs=inputs,
            system_prompt=scenario.system_prompt,
            user_prompt=scenario.user_prompt,
            model=model,
        )
        if inspect.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = await asyncio.to_thread(func, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, dict):
            raise TypeError(f"{scenario.entrypoint} must return a dict matching output_schema")
        return scenario.output_schema.model_validate(result).model_dump()

    graph = StateGraph(state_schema=state_schema, output_schema=scenario.output_schema)
    graph.add_node("agent_adapter", invoke)
    graph.add_edge(START, "agent_adapter")
    graph.add_edge("agent_adapter", END)
    return graph
