"""Normalize TOOL span attributes for the UiPath evaluator helpers.

Convert serialized ToolMessage payloads in ``output.value`` into the top-level
``{"content": ...}`` form consumed by the SDK's tool-output evaluators. Tool spans are
ordered by start time while other spans keep their positions.

Pass spans through :func:`normalize_tool_spans` before
``WorkloadExecution(workload_trace=...)``; the testbed and the GEPA runners do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan


def _clone_with_attrs(span: ReadableSpan, attrs: dict[str, Any]) -> ReadableSpan:
    """Copy ``span`` with ``attrs`` as its attributes (an ended span's attributes are sealed)."""
    return ReadableSpan(
        name=span.name,
        context=span.get_span_context(),
        parent=span.parent,
        resource=span.resource,
        attributes=attrs,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=getattr(span, "instrumentation_scope", None),
    )


def _flatten_tool_output(raw: Any) -> str:
    """Return the ``content`` of an ``output.value`` (JSON string or dict), else ``str(raw)``."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    else:
        parsed = raw

    if isinstance(parsed, dict):
        # langchain ToolMessage shape: {"type": "tool", "data": {"content": ...}}
        data = parsed.get("data")
        if isinstance(data, dict) and "content" in data:
            return str(data["content"])
        # Already-flat shape: {"content": ...}
        if "content" in parsed:
            return str(parsed["content"])

    return str(raw)


def _is_tool_span(span: ReadableSpan) -> bool:
    try:
        return bool((getattr(span, "attributes", {}) or {}).get("tool.name"))
    except Exception:
        return False


def _tools_in_start_order(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """``spans`` with the TOOL spans reordered by start time among their own positions.

    Spans are exported as they finish, so parallel tool calls can arrive out of order. Every
    other span keeps its place.
    """
    positions = [i for i, span in enumerate(spans) if _is_tool_span(span)]
    tools = sorted((spans[i] for i in positions), key=lambda s: getattr(s, "start_time", None) or 0)
    ordered = list(spans)
    for i, span in zip(positions, tools, strict=True):
        ordered[i] = span
    return ordered


def normalize_tool_spans(spans: Sequence[ReadableSpan]) -> list[ReadableSpan]:
    """Return ``spans`` with TOOL output values flattened to ``{"content": ...}`` and the TOOL
    spans in start order (every evaluator path - testbed, cached results, GEPA rollouts - calls
    this).

    Non-TOOL spans are returned unchanged and in place. The originals are not mutated.
    """
    normalized: list[ReadableSpan] = []
    for span in _tools_in_start_order(spans):
        try:
            attrs = dict(getattr(span, "attributes", {}) or {})
        except Exception:
            normalized.append(span)
            continue

        if not attrs.get("tool.name"):
            normalized.append(span)
            continue

        raw_output = attrs.get("output.value")
        if raw_output in (None, ""):
            normalized.append(span)
            continue

        flat = _flatten_tool_output(raw_output)
        # Re-emit as JSON so the evaluator's json.loads + dict path picks
        # up "content" at the top level.
        attrs["output.value"] = json.dumps({"content": flat})
        normalized.append(_clone_with_attrs(span, attrs))

    return normalized
