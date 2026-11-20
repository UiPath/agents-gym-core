"""Local LLM-call logging for GEPA-DSPy runs (langsmith-style, on-disk).

Two paths to capture:

1.  **DSPy reflection LM** calls go through litellm — invisible to LangChain
    instrumentation. We hook them via ``dspy.utils.callback.BaseCallback``'s
    ``on_lm_start`` / ``on_lm_end`` and stream JSONL to disk.

2.  **Agent rollouts** go through LangChain/LangGraph and are already captured
    by ``LangChainInstrumentor`` as OTEL spans. We post-process those spans
    after each rollout and dump LLM-kind ones to JSONL.

Both files live under ``<log_dir>/llm_calls/``:
  - ``reflection.jsonl`` — proposer LM calls
  - ``rollouts.jsonl``   — agent LM calls (from OTEL spans)

Each line is a self-contained JSON record so the files can be tailed and
parsed incrementally during a long-running run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — dspy is an optional dep
try:
    from dspy.utils.callback import BaseCallback as _DSPyBaseCallback
except Exception:  # pragma: no cover
    _DSPyBaseCallback = object  # type: ignore[misc,assignment]


class ReflectionCallLogger(_DSPyBaseCallback):  # type: ignore[misc]
    """Streaming JSONL logger for DSPy reflection-LM calls.

    Use as ``dspy.settings.configure(callbacks=[ReflectionCallLogger(path)])``
    or pass to :func:`agents_gym_core.prompt_optimization.gepa_dspy.optimize.optimize_dspy` via the wired
    log_dir. Writes one JSON record per LM call (start+end paired by call_id).
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._starts: dict[str, dict[str, Any]] = {}

    def on_lm_start(self, call_id: str, instance: Any, inputs: Any) -> None:
        with self._lock:
            self._starts[call_id] = {
                "call_id": call_id,
                "ts_start": time.time(),
                "model": getattr(instance, "model", None),
                "inputs": _safe_repr(inputs),
            }

    def on_lm_end(self, call_id: str, outputs: Any, exception: Any) -> None:
        with self._lock:
            start = self._starts.pop(call_id, None)
            record: dict[str, Any] = {
                "call_id": call_id,
                "ts_end": time.time(),
                "outputs": _safe_repr(outputs) if outputs is not None else None,
                "exception": (
                    {"type": type(exception).__name__, "msg": str(exception)}
                    if exception is not None
                    else None
                ),
            }
            if start is not None:
                record.update(
                    {
                        "ts_start": start["ts_start"],
                        "model": start["model"],
                        "inputs": start["inputs"],
                        "duration_s": record["ts_end"] - start["ts_start"],
                    }
                )
            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("ReflectionCallLogger failed to write: %s", e)


def dump_rollout_spans(spans: Sequence[Any], path: Path, datapoint: str) -> None:
    """Append LLM-kind OTEL spans from one rollout to ``rollouts.jsonl``.

    Each line is one LLM call record extracted from a span; tool spans are
    skipped (they're already captured separately for evaluator analysis).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for s in spans:
            try:
                attrs = dict(getattr(s, "attributes", {}) or {})
            except Exception:
                continue
            kind = attrs.get("openinference.span.kind")
            if kind != "LLM":
                continue
            ctx = s.get_span_context() if hasattr(s, "get_span_context") else None
            record = {
                "datapoint": datapoint,
                "name": getattr(s, "name", None),
                "trace_id": f"{ctx.trace_id:032x}" if ctx else None,
                "span_id": f"{ctx.span_id:016x}" if ctx else None,
                "start_ns": getattr(s, "start_time", None),
                "end_ns": getattr(s, "end_time", None),
                "duration_s": (
                    (s.end_time - s.start_time) / 1e9
                    if getattr(s, "end_time", None) and getattr(s, "start_time", None)
                    else None
                ),
                "model": attrs.get("llm.model_name") or attrs.get("model.name"),
                "input_value": attrs.get("input.value") or attrs.get("llm.input_messages"),
                "output_value": attrs.get("output.value") or attrs.get("llm.output_messages"),
                "token_count_prompt": attrs.get("llm.token_count.prompt"),
                "token_count_completion": attrs.get("llm.token_count.completion"),
            }
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def _safe_repr(obj: Any, max_chars: int = 50000) -> Any:
    """Convert obj to a json-serializable form, truncating very long strings."""
    try:
        json.dumps(obj, default=str)
        # Already serializable — but truncate long string fields
        return _truncate(obj, max_chars)
    except Exception:
        s = str(obj)
        if len(s) > max_chars:
            s = s[:max_chars] + "...[truncated]"
        return s


def _truncate(obj: Any, max_chars: int) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= max_chars else obj[:max_chars] + "...[truncated]"
    if isinstance(obj, dict):
        return {k: _truncate(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, max_chars) for v in obj]
    return obj
