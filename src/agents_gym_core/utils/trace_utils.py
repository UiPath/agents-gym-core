"""OpenTelemetry trace and log collection for agent evaluation.

:func:`setup_tracer` instruments LangChain and collects spans (per trace) and Python logs (via
a ``LoggingHandler`` on the root logger) in memory. :func:`save_execution_data` writes the
spans evaluators need to each datapoint's ``output.json``.
"""

import atexit
import base64
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from google.protobuf import json_format
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.common import _internal as _otlp_internal
from opentelemetry.exporter.otlp.proto.common import trace_encoder
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind

OUTPUT_FILENAME = "output.json"
AGENT_OUTPUT_FILENAME = "agent_output.json"


class PerTraceSpanExporter(SpanExporter):
    """Span exporter that buckets finished spans by trace_id.

    ``pop_trace`` returns and frees one trace's spans in O(1); filtering the flat list of an
    ``InMemorySpanExporter`` over thousands of spans blocks the event loop.
    """

    def __init__(self) -> None:
        self._spans: dict[int, list[ReadableSpan]] = defaultdict(list)
        self._lock = threading.Lock()
        self._closed = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            if self._closed:  # a shut-down provider's spans are dropped, not kept forever
                return SpanExportResult.SUCCESS
            for span in spans:
                ctx = span.get_span_context()
                if ctx is not None:
                    self._spans[ctx.trace_id].append(span)
        return SpanExportResult.SUCCESS

    def pop_trace(self, trace_id: int) -> list[ReadableSpan]:
        """Return and remove all spans for the given trace_id."""
        with self._lock:
            return self._spans.pop(trace_id, [])

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._spans.clear()


_original_encode_value = getattr(
    _otlp_internal._encode_value, "__wrapped__", _otlp_internal._encode_value
)


@wraps(_original_encode_value)
def _patched_encode_value(value: Any, *args: Any, **kwargs: Any) -> Any:
    """``_encode_value`` that encodes integers outside int64 (e.g. trace IDs) as hex strings.

    Forward positional and keyword arguments to the installed OpenTelemetry encoder,
    preserving its call signature across supported SDK versions.
    """
    if isinstance(value, int) and (value > 2**63 - 1 or value < -(2**63)):
        return _otlp_internal.PB2AnyValue(string_value=f"0x{value:032x}")  # pyright: ignore[reportPrivateImportUsage]
    return _original_encode_value(value, *args, **kwargs)


_otlp_internal._encode_value = _patched_encode_value

_instrumentor_instance: LangChainInstrumentor | None = None
_is_instrumented: bool = False
_logger_provider: LoggerProvider | None = None
_tracer_provider: TracerProvider | None = None
_logging_handler: LoggingHandler | None = None
_shutdown_registered: bool = False
_session: "tuple[PerTraceSpanExporter, TracerProvider, InMemoryLogExporter] | None" = None


def info(message: str, logger_name: str = "agents_gym_core", **attributes: Any) -> None:
    """Log at INFO level, passing ``attributes`` as the ``otel.attributes`` extra."""
    logger = logging.getLogger(logger_name)
    extra = {"otel.attributes": attributes} if attributes else {}
    logger.info(message, extra=extra)


def warning(message: str, logger_name: str = "agents_gym_core", **attributes: Any) -> None:
    """Log at WARNING level, passing ``attributes`` as the ``otel.attributes`` extra."""
    logger = logging.getLogger(logger_name)
    extra = {"otel.attributes": attributes} if attributes else {}
    logger.warning(message, extra=extra)


def error(message: str, logger_name: str = "agents_gym_core", **attributes: Any) -> None:
    """Log at ERROR level, passing ``attributes`` as the ``otel.attributes`` extra."""
    logger = logging.getLogger(logger_name)
    extra = {"otel.attributes": attributes} if attributes else {}
    logger.error(message, extra=extra)


class InMemoryLogExporter(LogRecordExporter):
    """Log exporter that keeps log records in memory."""

    def __init__(self):
        super().__init__()
        self._logs: list[Any] = []
        self._processor: BatchLogRecordProcessor | None = None
        self._logger_provider: LoggerProvider | None = None

    def export(self, batch: Sequence[Any]) -> LogRecordExportResult:
        self._logs.extend(batch)
        return LogRecordExportResult.SUCCESS

    def get_logs(self) -> list[Any]:
        """Flush the logger provider and batch processor, then return every collected record."""
        if hasattr(self, "_logger_provider") and self._logger_provider is not None:
            self._logger_provider.force_flush()
        if self._processor is not None:
            self._processor.force_flush()
        return self._logs

    def clear(self) -> None:
        self._logs.clear()

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _shutdown_otel() -> None:
    """Shut down the OpenTelemetry providers before interpreter shutdown."""
    global _logger_provider, _tracer_provider, _logging_handler

    # Remove the handler first so logging.shutdown() does not flush it.
    if _logging_handler is not None:
        try:
            logging.getLogger().removeHandler(_logging_handler)
            _logging_handler.close()
        except Exception:
            pass

    # Shutting the logger provider down here keeps it from creating threads at shutdown.
    if _logger_provider is not None:
        try:
            _logger_provider.shutdown()
        except Exception:
            pass

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception:
            pass


def setup_tracer(
    experiment_dir: Path | None = None,
) -> tuple[PerTraceSpanExporter, TracerProvider, InMemoryLogExporter]:
    """Install in-memory span and log exporters and instrument LangChain.

    Each call shuts down and replaces the providers, root-logger handler and instrumentor of
    the previous call. ``experiment_dir`` is unused.

    Returns:
        The span exporter (read with ``pop_trace(trace_id)``), the tracer provider and the
        log exporter.
    """
    global \
        _instrumentor_instance, \
        _is_instrumented, \
        _logger_provider, \
        _tracer_provider, \
        _logging_handler, \
        _shutdown_registered

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception:
            pass
    if _logger_provider is not None:
        try:
            _logger_provider.shutdown()
        except Exception:
            pass

    span_exporter = PerTraceSpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer_provider = tracer_provider

    log_exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    log_processor = BatchLogRecordProcessor(log_exporter)
    logger_provider.add_log_record_processor(log_processor)
    _logs.set_logger_provider(logger_provider)
    _logger_provider = logger_provider

    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)

    original_flush = handler.flush

    def safe_flush():
        """Flush, ignoring the "can't create new thread" error raised at interpreter shutdown."""
        try:
            original_flush()
        except RuntimeError as e:
            if "can't create new thread" not in str(e):
                raise

    handler.flush = safe_flush

    if _logging_handler is not None:
        try:
            logging.getLogger().removeHandler(_logging_handler)
            _logging_handler.close()
        except Exception:
            pass

    # On the root logger, so records from every logger reach OpenTelemetry.
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    _logging_handler = handler

    # atexit runs handlers last-in first-out, so this runs before logging's own shutdown.
    if not _shutdown_registered:
        atexit.register(_shutdown_otel)
        _shutdown_registered = True

    # Re-instrument so LangChain spans go to the new tracer provider.
    if _is_instrumented and _instrumentor_instance is not None:
        try:
            _instrumentor_instance.uninstrument()
        except Exception:
            pass

    _instrumentor_instance = LangChainInstrumentor()
    _instrumentor_instance.instrument(tracer_provider=tracer_provider)
    _is_instrumented = True

    log_exporter._processor = log_processor
    log_exporter._logger_provider = logger_provider

    global _session
    _session = (span_exporter, tracer_provider, log_exporter)
    return _session


def current_tracing_session() -> (
    tuple[PerTraceSpanExporter, TracerProvider, InMemoryLogExporter] | None
):
    """The session of the last ``setup_tracer()`` call while it is still exporting, else None."""
    if _session is None or _session[0]._closed:
        return None
    return _session


def _is_relevant_span(span: Any) -> bool:
    """Whether ``output.json`` keeps the span.

    Keeps tool spans (read by trace-based evaluators), LangGraph/terminate chain spans (the
    agent output) and LLM spans (slimmed later; model, tokens and tool-call names).
    """
    if not hasattr(span, "attributes") or not span.attributes:
        return False
    attrs = dict(span.attributes)
    if attrs.get("tool.name") is not None:
        return True
    span_kind = attrs.get("openinference.span.kind")
    if getattr(span, "name", "") in ("LangGraph", "terminate") and span_kind == "CHAIN":
        return True
    if span_kind == "LLM":
        return True
    return False


# Dropped from every span: large and unused by evaluators.
_DROP_ATTRIBUTES = {
    "input.mime_type",
    "output.mime_type",
    "metadata",
}

# Also dropped from LLM spans, by key prefix.
_LLM_DROP_PREFIXES = (
    "llm.tools.",  # full tool schemas repeated on every LLM call
    "llm.invocation_parameters",  # model config repeated on every LLM call
    "llm.input_messages.",  # full prompt history per call
    "llm.output_messages.",  # full response per call
    "llm.token_count.completion_details.",  # granular breakdowns (audio, reasoning)
    "llm.token_count.prompt_details.",  # granular breakdowns (audio, cache_read)
    "llm.system",  # redundant with llm.provider
)


def _strip_otlp_attributes(otlp_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove selected attributes and metadata from serialized OTLP JSON.

    All spans lose ``_DROP_ATTRIBUTES``, chain spans also lose ``input.value`` (the whole
    conversation history), and LLM spans lose ``_LLM_DROP_PREFIXES``. Resource attributes,
    instrumentation scope and span flags are omitted.
    """
    for resource_span in otlp_dict.get("resource_spans", []):
        if "resource" in resource_span:
            resource_span["resource"].pop("attributes", None)
        for scope_span in resource_span.get("scope_spans", []):
            scope_span.pop("scope", None)
            for span_data in scope_span.get("spans", []):
                span_data.pop("flags", None)

                attrs = span_data.get("attributes", [])
                if not attrs:
                    continue

                span_kind = None
                for attr in attrs:
                    if attr.get("key") == "openinference.span.kind":
                        span_kind = attr.get("value", {}).get("string_value")
                        break

                filtered_attrs = []
                for attr in attrs:
                    key = attr.get("key", "")

                    if key in _DROP_ATTRIBUTES:
                        continue

                    if span_kind == "CHAIN" and key == "input.value":
                        continue

                    if span_kind == "LLM" and any(key.startswith(p) for p in _LLM_DROP_PREFIXES):
                        continue

                    filtered_attrs.append(attr)

                span_data["attributes"] = filtered_attrs

    return otlp_dict


def save_execution_data(datapoint_dir: Path, spans: Sequence[Any]) -> None:
    """Write the relevant spans to ``datapoint_dir/output.json`` as a JSON list of OTLP objects.

    See :func:`_is_relevant_span` and :func:`_strip_otlp_attributes`. Nothing is written when
    no span is relevant. :func:`load_execution_data` reads the file back.
    """
    if not spans:
        return

    relevant_spans = [s for s in spans if _is_relevant_span(s)]
    if not relevant_spans:
        return

    datapoint_dir.mkdir(parents=True, exist_ok=True)
    output_file = datapoint_dir / OUTPUT_FILENAME

    stripped_objects = []
    for span in relevant_spans:
        pb_span = trace_encoder.encode_spans([span])
        json_str = json_format.MessageToJson(pb_span, preserving_proto_field_name=True, indent=2)
        otlp_dict = json.loads(json_str)
        stripped_objects.append(_strip_otlp_attributes(otlp_dict))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stripped_objects, f, indent=2)
        f.write("\n")


def save_agent_output(datapoint_dir: Path, agent_output: Any) -> None:
    """Write ``{"agent_output": agent_output}`` to ``datapoint_dir/agent_output.json``.

    A run loaded from disk is then scored against the output the agent returned, an error dict
    included, rather than the one :func:`get_agent_output_from_trace` parses back. An output that
    is not JSON is not saved, and a file left by an earlier run is removed, so the run is loaded
    from its trace.
    """
    path = datapoint_dir / AGENT_OUTPUT_FILENAME
    try:
        text = json.dumps({"agent_output": agent_output}, indent=2)
    except (TypeError, ValueError):
        path.unlink(missing_ok=True)
        return
    datapoint_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_agent_output(datapoint_dir: Path) -> dict[str, Any] | None:
    """``{"agent_output": ...}`` as :func:`save_agent_output` wrote it; None if absent or
    unreadable."""
    try:
        saved = json.loads((datapoint_dir / AGENT_OUTPUT_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return saved if isinstance(saved, dict) and "agent_output" in saved else None


def get_agent_output_from_trace(trace: list[ReadableSpan]) -> dict[str, Any] | str:
    """Return the agent output recorded in ``trace``, or ``{}`` if there is none.

    See :func:`_get_output_from_langgraph_span`.
    """
    if not trace:
        return {}

    return _get_output_from_langgraph_span(trace) or {}


def _get_output_from_langgraph_span(trace: list[ReadableSpan]) -> dict[str, Any] | str | None:
    """Return ``output.value`` of the last LangGraph/terminate CHAIN span that has one.

    A JSON object is returned as a dict and anything else as the stripped string: the
    openinference tracer records a single-key dict with a string value
    (``{"result": "answer"}``) as the bare string, and that string may itself parse as
    JSON (``"17"``, ``"true"``), in which case it is still the agent's answer text.
    """
    for span in reversed(trace):
        attrs = dict(span.attributes) if hasattr(span, "attributes") and span.attributes else {}
        if (
            span.name in ("LangGraph", "terminate")
            and attrs.get("openinference.span.kind") == "CHAIN"
        ):
            raw = attrs.get("output.value")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
                if raw_stripped := raw.strip():
                    return raw_stripped
    return None


def load_execution_data(datapoint_dir: Path) -> list[ReadableSpan]:
    """Read ``datapoint_dir/output.json`` back as ReadableSpans; ``[]`` if missing or empty.

    A file that cannot be read or parsed is logged as a warning and also yields ``[]``.
    """
    output_file = datapoint_dir / OUTPUT_FILENAME

    if not output_file.exists():
        return []

    try:
        spans: list[ReadableSpan] = []

        with open(output_file, encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return []

        # A JSON list of OTLP objects (as written by save_execution_data), or a single object.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for otlp_data in parsed:
                    spans.extend(_extract_spans_from_otlp(otlp_data))
            else:
                spans.extend(_extract_spans_from_otlp(parsed))
        except json.JSONDecodeError:
            # Concatenated OTLP objects, which are not one JSON document.
            for json_str in _split_json_objects(content):
                if not json_str.strip():
                    continue
                otlp_data = json.loads(json_str)
                spans.extend(_extract_spans_from_otlp(otlp_data))

        return spans
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        warning(
            f"Failed to load output.json: {e}",
            datapoint_dir=str(datapoint_dir),
        )
        return []


def _split_json_objects(content: str) -> list[str]:
    """Split concatenated, possibly multi-line JSON objects by counting braces outside strings."""
    objects: list[str] = []
    depth = 0
    start = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(content[start : i + 1])

    return objects


def _extract_spans_from_otlp(otlp_data: dict[str, Any]) -> list[ReadableSpan]:
    """Build a ReadableSpan for every span under ``otlp_data["resource_spans"]``."""
    spans: list[ReadableSpan] = []

    for resource_span in otlp_data.get("resource_spans", []):
        for scope_span in resource_span.get("scope_spans", []):
            for span_data in scope_span.get("spans", []):
                span = _create_readable_span_from_otlp(span_data)
                spans.append(span)

    return spans


def _create_readable_span_from_otlp(span_data: dict[str, Any]) -> ReadableSpan:
    """Rebuild a ReadableSpan's name, trace and span IDs, attributes and timestamps from an
    OTLP span dict. The parent is not restored, and the kind is always INTERNAL.
    """
    attributes: dict[str, Any] = {}
    for attr in span_data.get("attributes", []):
        key = attr.get("key", "")
        value_obj = attr.get("value", {})

        if "string_value" in value_obj:
            attributes[key] = value_obj["string_value"]
        elif "int_value" in value_obj:
            attributes[key] = int(value_obj["int_value"])
        elif "double_value" in value_obj:
            attributes[key] = float(value_obj["double_value"])
        elif "bool_value" in value_obj:
            attributes[key] = value_obj["bool_value"]
        elif "array_value" in value_obj:
            array_values = []
            for item in value_obj["array_value"].get("values", []):
                if "string_value" in item:
                    array_values.append(item["string_value"])
                elif "int_value" in item:
                    array_values.append(int(item["int_value"]))
                elif "double_value" in item:
                    array_values.append(float(item["double_value"]))
                elif "bool_value" in item:
                    array_values.append(item["bool_value"])
            attributes[key] = array_values

    start_time = None
    end_time = None
    if "start_time_unix_nano" in span_data:
        start_time = int(span_data["start_time_unix_nano"])
    if "end_time_unix_nano" in span_data:
        end_time = int(span_data["end_time_unix_nano"])

    # The saved IDs (base64), which the trajectory judges use to tell tool calls apart.
    context = None
    try:
        trace_id, span_id = (
            int.from_bytes(base64.b64decode(span_data[key]), "big")
            for key in ("trace_id", "span_id")
        )
        context = SpanContext(trace_id, span_id, is_remote=False)
    except (KeyError, TypeError, ValueError):
        pass

    return ReadableSpan(
        name=span_data.get("name", ""),
        context=context,
        attributes=attributes,
        start_time=start_time,
        end_time=end_time,
        kind=SpanKind.INTERNAL,
    )
