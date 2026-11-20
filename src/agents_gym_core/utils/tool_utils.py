"""Utility functions for working with tools."""

import json
from typing import Any

import yaml
from langchain_core.tools import BaseTool


def tool_to_dict(tool: BaseTool) -> dict[str, Any]:
    """Return the tool's name and description as a dict."""
    return {"name": tool.name, "description": tool.description}


def tool_to_str(tool: BaseTool, as_yaml: bool = False, include_tags: bool = False) -> str:
    """Render the tool's name and description as JSON or YAML, optionally in ``<tool>`` tags."""
    tool_repr = (
        yaml.dump(tool_to_dict(tool), sort_keys=False)
        if as_yaml
        else json.dumps(tool_to_dict(tool), sort_keys=False)
    )
    if include_tags:
        return f"<tool>\n{tool_repr}\n</tool>"
    return tool_repr
