"""GEPA-DSPy: the paper's GEPA baseline for the enterprise agents.

Uses upstream ``dspy.GEPA`` (PyPI ``dspy>=3.0``). The agent is wrapped as a one-predictor
:class:`dspy.Module` whose ``signature.instructions`` is the agent's ``system_prompt``, the
component GEPA mutates. ``forward()`` runs the agent graph (``basic`` loop); the metric runs
the agent's evaluators.

The native GEPA variant in ``agents_gym_core.prompt_optimization.gepa`` uses LangChain models
directly.
"""

from agents_gym_core.prompt_optimization.gepa_dspy.optimize import optimize_dspy

__all__ = ["optimize_dspy"]
