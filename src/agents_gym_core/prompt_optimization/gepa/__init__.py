"""Native GEPA variant with greedy candidate selection.

Reflective prompt optimization adapted from gepa-artifact to operate directly on Gym
primitives. The paper's enterprise GEPA baseline uses upstream ``dspy.GEPA`` through
``gepa_dspy``.

A *candidate* is a ``dict[str, str]`` mapping component name to instruction
text. For ``basic`` loops there is exactly one component:
``system_prompt``. The algorithm runs each candidate via the gym's
``get_loop`` factory, scores it with the agent's existing evaluator
battery, and asks an LLM proposer to mutate the instruction based on
per-evaluator feedback.
"""

from agents_gym_core.prompt_optimization.gepa.optimize import OptimizeConfig, optimize
from agents_gym_core.prompt_optimization.gepa.rollout import Candidate, RolloutResult
from agents_gym_core.prompt_optimization.gepa.state import OptimizationState

__all__ = [
    "Candidate",
    "OptimizationState",
    "OptimizeConfig",
    "RolloutResult",
    "optimize",
]
