"""Shared text-parsing helpers used across loops, evaluators, and tools."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, stripping markdown fences.

    Tries full-text parse first, then falls back to extracting the outermost
    ``{...}`` substring. Returns ``None`` if no valid JSON object is found.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def content_to_text(content: Any) -> str:
    """Extract plain text from an LLM response content field.

    Handles string content, multimodal content lists (with ``text`` or
    ``content`` dict keys), and falls back to ``str()`` for anything else.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(parts)
    return str(content)
