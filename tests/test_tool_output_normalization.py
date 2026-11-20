"""Tool-call outputs reach the evaluators on the testbed path.

LangChain TOOL spans carry the whole ToolMessage in ``output.value``
(``{"type": "tool", "data": {"content": ...}}``), while the UiPath extractor behind
``ToolCallOutputEvaluator`` only reads a top-level ``content``; loading a trace normalizes it.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from uipath.eval._helpers.evaluators_helpers import extract_tool_calls_outputs
from uipath.eval.models.evaluation_set import EvaluationItem

from agents_gym_core.runtime.run_utils import get_datapoint_dir, load_execution
from agents_gym_core.utils import DirectoryProvider
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import save_execution_data

TOOL_MESSAGE = {
    "type": "tool",
    "data": {"content": "[('Rock',)]", "type": "tool", "name": "sql_db_query", "status": "success"},
}


def _tool_spans() -> list:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span(
        "sql_db_query",
        attributes={
            "openinference.span.kind": "TOOL",
            "tool.name": "sql_db_query",
            "input.value": json.dumps({"query": "SELECT 1"}),
            "output.value": json.dumps(TOOL_MESSAGE),
        },
    ):
        pass
    return list(exporter.get_finished_spans())


def test_a_saved_trace_gives_the_evaluators_the_tool_output(tmp_path: Path) -> None:
    datapoint = EvaluationItem.model_construct(name="dp", inputs={}, input_mocking_strategy=None)
    save_execution_data(get_datapoint_dir(tmp_path, "dp", 1), _tool_spans())

    execution = load_execution(
        tmp_path,
        datapoint,
        cast(DirectoryProvider, None),
        cast(Agent, SimpleNamespace(name="sql", version="test")),
        run_number=1,
    )

    assert execution is not None
    outputs = extract_tool_calls_outputs(execution.workload_execution.workload_trace)
    assert [(o.name, o.output) for o in outputs] == [("sql_db_query", "[('Rock',)]")]
