"""Instantiator for tools from configuration."""

import inspect
import textwrap
from copy import deepcopy
from functools import partial
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import get_object, instantiate
from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from omegaconf import DictConfig, OmegaConf, read_write


class ToolInstantiator:
    """Instantiator for tools from configuration."""

    @staticmethod
    def _wrap_as_tool(obj: Any) -> BaseTool:
        """Return a tool or wrap a callable without executing it."""
        if isinstance(obj, BaseTool):
            return obj

        if callable(obj):
            return StructuredTool.from_function(obj)

        raise TypeError(f"Cannot convert {type(obj)} to BaseTool")

    @staticmethod
    def _copy_configured_tool(target: BaseTool, tool_cfg: DictConfig) -> BaseTool:
        """Apply version-local fields without invoking the shared decorated tool."""
        reserved = {"_target_", "_partial_", "_recursive_", "_convert_"}
        keys = {str(key) for key in tool_cfg if key not in reserved}
        unknown = sorted(keys - set(type(target).model_fields))
        if unknown:
            raise TypeError(f"not fields of tool '{target.name}': {unknown}")
        overrides: dict[str, Any] = {}
        for key in keys:
            value = tool_cfg[key]
            overrides[key] = (
                OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            )
        return target.model_copy(update=overrides) if overrides else target

    @staticmethod
    def _instantiate_single_tool(tool_cfg: DictConfig) -> BaseTool:
        """Resolve existing tools, or instantiate a factory exactly once through Hydra.

        Existing tools retain version-local field overrides. Factory targets are captured
        once before nested argument conversion; deferred factories are not runtime tools.
        Hydra target restrictions and constructor failures propagate without a fallback.
        """
        target_name = str(tool_cfg._target_)
        if "${" in target_name:
            raise InstantiationException(
                "Tool _target_ must resolve to a literal import path, not escaped "
                "or unresolved interpolation syntax"
            )
        tool_cfg = deepcopy(tool_cfg)
        with read_write(tool_cfg):
            tool_cfg._target_ = target_name
        try:
            # Authorize the target without executing it or its nested arguments.
            authorized = instantiate({"_target_": target_name, "_partial_": True})
        except InstantiationException as error:
            # Only non-callable tool instances need lookup after Hydra authorization.
            if str(error).startswith("Expected a callable target,"):
                target = get_object(target_name)
                if isinstance(target, BaseTool):
                    return ToolInstantiator._copy_configured_tool(target, tool_cfg)
            raise
        if not isinstance(authorized, partial):
            raise TypeError("Hydra target resolution must return a partial")
        if isinstance(authorized.func, BaseTool):
            return ToolInstantiator._copy_configured_tool(authorized.func, tool_cfg)
        if tool_cfg.get("_partial_", False) is True:
            raise InstantiationException(
                "Tool factories do not support _partial_=True; configure a factory "
                "that returns a BaseTool or callable instead"
            )
        result = instantiate(tool_cfg, _convert_=tool_cfg.get("_convert_", "all"))
        return ToolInstantiator._wrap_as_tool(result)

    @staticmethod
    def _docstring_description(tool: BaseTool) -> BaseTool:
        """Return ``tool`` with a docstring-derived description in Python 3.13's form.

        Python 3.13 removes docstring indentation at compile time; 3.11 and 3.12 keep it, so
        the ``Args:`` block of a description taken from a docstring would reach the model
        indented differently. Descriptions set in configuration are left as they are.
        """
        function = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
        doc = getattr(function, "__doc__", None)
        if doc and tool.description == textwrap.dedent(doc).strip():
            description = inspect.cleandoc(doc)
            if description != tool.description:
                return tool.model_copy(update={"description": description})
        return tool

    @staticmethod
    def instantiate_tools(agent_cfg: DictConfig) -> list[BaseTool]:
        """Instantiate every tool listed in ``agent_cfg.tools`` (``agent_cfg`` is ``cfg.agent``).

        Raises:
            TypeError: If a tool cannot be converted to BaseTool.
        """
        tools = []
        if not hasattr(agent_cfg, "tools") or not agent_cfg.tools:
            return tools

        tools_list = agent_cfg.tools

        if not tools_list:
            return tools

        for tool_cfg in tools_list:
            tool = ToolInstantiator._instantiate_single_tool(tool_cfg)
            tools.append(ToolInstantiator._docstring_description(tool))

        return tools
