"""Loop templates for agent execution."""

from agents_gym_core.loops.basic_loop import (
    BasicLoop,
    BasicLoopAny,
    BasicLoopAnyGraceful,
    BasicLoopAutoGraceful,
)
from agents_gym_core.loops.deep_agents_loop import (
    build_deep_agents_input,
    create_deep_agents_graph,
    create_deep_agents_uncompiled_graph,
    normalize_deep_agents_output,
)
from agents_gym_core.loops.dspy_program_loop import (
    create_dspy_program_graph,
    create_dspy_program_uncompiled_graph,
)
from agents_gym_core.loops.loop_factory import get_loop, get_uncompiled_loop
from agents_gym_core.loops.planning_loop import (
    PlanningLoop,
    PlanningSystemPlanLoop,
    PlanningUserPlanLoop,
)
from agents_gym_core.loops.summarizing_loop import (
    SummarizingLoop,
    SummarizingLoopAny,
)
from agents_gym_core.loops.types import (
    DEFAULT_LOOP_VERSION,
    ExtendedGraphNode,
    LoopDetails,
    LoopName,
)
from agents_gym_core.loops.utils import build_dspy_output_model, build_dspy_state_model
from agents_gym_core.loops.validation_loop import ValidationLoop

__all__ = [
    "BasicLoop",
    "BasicLoopAny",
    "BasicLoopAutoGraceful",
    "BasicLoopAnyGraceful",
    "PlanningLoop",
    "PlanningSystemPlanLoop",
    "PlanningUserPlanLoop",
    "ValidationLoop",
    "SummarizingLoop",
    "SummarizingLoopAny",
    "create_deep_agents_graph",
    "create_deep_agents_uncompiled_graph",
    "build_deep_agents_input",
    "normalize_deep_agents_output",
    "create_dspy_program_graph",
    "create_dspy_program_uncompiled_graph",
    "build_dspy_state_model",
    "build_dspy_output_model",
    "LoopName",
    "LoopDetails",
    "ExtendedGraphNode",
    "DEFAULT_LOOP_VERSION",
    "get_loop",
    "get_uncompiled_loop",
]
