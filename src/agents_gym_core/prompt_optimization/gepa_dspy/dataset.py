"""Convert a gym EvaluationSet to a list of dspy.Example.

DSPy's GEPA passes ``gold`` (the example) into the metric callable. We stash
the original ``EvaluationItem`` in a private ``_eval_item`` field so the
metric can reach the per-evaluator criteria without re-loading the eval set.
"""

from __future__ import annotations

import dspy
from uipath.eval.models.evaluation_set import EvaluationItem

from agents_gym_core.utils.gym_types import Agent


def evalset_to_dspy_examples(agent: Agent) -> list[dspy.Example]:
    """One ``dspy.Example`` per gym datapoint, inputs matching the gym schema."""
    input_fields = list(agent.input_schema.model_fields.keys())
    examples: list[dspy.Example] = []
    for item in agent.evaluation_set.evaluations or []:
        fields: dict = {f: item.inputs.get(f) for f in input_fields}
        fields["_eval_item"] = item
        ex = dspy.Example(**fields).with_inputs(*input_fields)
        examples.append(ex)
    return examples


def get_eval_item(example: dspy.Example) -> EvaluationItem:
    return example._eval_item  # type: ignore[attr-defined]
