"""Wrap a gym Agent as a single-predictor dspy.Module.

The wrapper carries one ``dspy.Predict`` whose
``signature.instructions`` IS the gym agent's ``system_prompt``. DSPy's GEPA
uses ``named_predictors()`` and ``signature.with_instructions(...)`` to evolve
that string. ``forward()`` ignores DSPy's internal predict logic and runs
the gym's compiled LangGraph instead.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from uuid import uuid4

import dspy
from langchain_core.language_models import BaseChatModel
from uipath_langchain.agent.exceptions import AgentRuntimeError

from agents_gym_core.loops import get_loop
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION

# One tracer setup per process, shared with native GEPA's rollouts (the benchmark's test
# evaluation): a second ``setup_tracer`` call would shut down the providers of the first.
from agents_gym_core.prompt_optimization.gepa.rollout import _ensure_tracing
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import PerTraceSpanExporter

# DSPy runs rollouts and evaluators from worker threads. They all use this one event loop,
# because an async model client stays on the loop that opened its connections. It is kept
# here and not on GymAgentModule, which DSPy copies for every candidate.
EVENT_LOOP = asyncio.new_event_loop()
threading.Thread(target=EVENT_LOOP.run_forever, name="gepa-dspy-loop", daemon=True).start()


def _build_signature(input_field_names: list[str], instructions: str) -> type[dspy.Signature]:
    """Build a dspy.Signature whose inputs match the gym agent's input_schema.

    The output field is a single opaque ``output`` payload (we hand back the
    gym graph's full result dict — the metric does any per-evaluator scoring).
    """
    fields = {name: dspy.InputField() for name in input_field_names}
    fields["output"] = dspy.OutputField()
    return dspy.make_signature(fields, instructions)


class GymAgentModule(dspy.Module):
    """One-predictor DSPy wrapper around a gym Agent.

    Attributes:
        predict: a ``dspy.Predict`` whose ``signature.instructions`` is the
            evolvable system prompt. This is the only thing DSPy's GEPA touches.
    """

    def __init__(
        self,
        gym_agent: Agent,
        *,
        model: BaseChatModel,
        loop_name: str = "basic",
        loop_version: str = DEFAULT_LOOP_VERSION,
    ):
        super().__init__()
        self._gym_agent = gym_agent
        self._model = model
        self._loop_name = loop_name
        self._loop_version = loop_version

        input_fields = list(gym_agent.input_schema.model_fields.keys())
        # dspy stores the instructions as inspect.cleandoc(system_prompt), and rollouts run
        # with that: compare an evolved prompt with it to tell a kept seed.
        sig = _build_signature(input_fields, gym_agent.system_prompt)
        self.predict = dspy.Predict(sig)

    @property
    def current_instructions(self) -> str:
        return self.predict.signature.instructions  # pyright: ignore[reportAttributeAccessIssue]

    def _run_gym_graph(self, inputs: dict[str, Any]) -> tuple[Any, list[Any]]:
        """Run the async gym graph from DSPy's worker thread, on the shared event loop."""
        return asyncio.run_coroutine_threadsafe(self._ainvoke(inputs), EVENT_LOOP).result()

    async def _ainvoke(self, inputs: dict[str, Any]) -> tuple[Any, list[Any]]:
        # Overlay the (possibly mutated) instruction onto the gym agent
        mutated_agent = self._gym_agent.model_copy(
            update={"system_prompt": self.current_instructions}
        )
        compiled = get_loop(
            loop_name=self._loop_name,
            version=self._loop_version,
            scenario=mutated_agent,
            models={"model": self._model},
        )

        state = _ensure_tracing()
        span_exporter: PerTraceSpanExporter = state["span_exporter"]
        tracer_provider = state["tracer_provider"]
        tracer = tracer_provider.get_tracer(__name__)

        # Cap on one rollout: dspy's Evaluate waits for every result of a minibatch, so a
        # rollout that never returned would hold up the whole optimization.
        rollout_timeout_s = 240.0

        with tracer.start_as_current_span(name="gepa_dspy_rollout") as root:
            trace_id = root.get_span_context().trace_id
            try:
                output = await asyncio.wait_for(
                    compiled.ainvoke(
                        inputs,
                        config={"configurable": {"thread_id": str(uuid4())}},
                    ),
                    timeout=rollout_timeout_s,
                )
            except TimeoutError:
                output = {
                    "type": "error",
                    "code": "GENERAL_EXECUTION_ERROR",
                    "title": "RolloutTimeout",
                    "details": f"exceeded {rollout_timeout_s}s",
                }
            except Exception as e:
                output = {
                    "type": "error",
                    "code": e.error_info.code
                    if isinstance(e, AgentRuntimeError)
                    else "GENERAL_EXECUTION_ERROR",
                    "title": type(e).__name__,
                    "details": str(e),
                }
        spans = span_exporter.pop_trace(trace_id)
        return output, spans

    def forward(self, **inputs: Any) -> dspy.Prediction:
        """Sync entry point used by dspy.Evaluate / dspy.GEPA.

        Note: dspy.Prediction inherits from dspy.Example, which has reserved
        attribute names (``inputs``, ``labels``, ``with_inputs``, …). We
        prefix our payload fields with ``agent_`` to avoid colliding with
        those bound methods.
        """
        output, spans = self._run_gym_graph(inputs)

        # dspy.GEPA builds its reflective dataset from predictor entries in
        # ``dspy.settings.trace`` (see dspy/teleprompt/gepa/gepa_utils.py:224).
        # forward() runs the agent graph rather than ``self.predict``, so it
        # records the predictor's entry itself, as ``Predict._forward_postprocess``
        # does, with the agent output as the prediction.
        if dspy.settings.trace is not None and dspy.settings.max_trace_size > 0:
            sig_output_fields = list(self.predict.signature.output_fields.keys())  # pyright: ignore[reportAttributeAccessIssue]
            output_field = sig_output_fields[0] if sig_output_fields else "output"
            output_str = output if isinstance(output, str) else str(output)
            prediction = dspy.Prediction(**{output_field: output_str})
            trace = dspy.settings.trace
            if len(trace) >= dspy.settings.max_trace_size:
                trace.pop(0)
            trace.append((self.predict, dict(inputs), prediction))

        return dspy.Prediction(
            agent_output=output,
            agent_spans=spans,
            agent_inputs=inputs,
        )
