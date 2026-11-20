"""DSPy program loop — runs any ``dspy.Module`` as a gym agent.

The module class is passed as the version slot of the loop spec:

    --loops dspy-program:gepa_artifact.benchmarks.hover.hover_program.HoverMultiHop
    --loops dspy-program:gepa_artifact.benchmarks.papillon.papillon_program.PAPILLON

The primary model (``"model"`` key in ``--models``) is injected via
``dspy.context(lm=lm)``.  The other models in the dict are wrapped as ``GymLM``
and passed to the constructor by parameter name; a required parameter with no
matching model gets the primary model (with a warning).  A module-level instance
is deep-copied.

``signature_prompts`` in ``agent.yaml`` override per-predictor instructions
at graph-build time, making the program optimizable by the ReAPO and GEPA
optimizers.
"""

import copy
import inspect
from importlib import import_module
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents_gym_core.gepa_artifact import skip_spacy_model_download
from agents_gym_core.loops.types import LoopDetails, LoopName
from agents_gym_core.loops.utils import build_dspy_output_model, build_dspy_state_model
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import info, warning


def _resolve_program(
    dotted_path: str,
    models: dict[str, Any],
) -> Any:
    """Import and return a ``dspy.Module`` from a dotted path.

    Class → constructor params are matched by name from ``models``.
    Instance → deep-copied.
    """
    import dspy  # pyright: ignore[reportMissingImports]

    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid dotted path {dotted_path!r}: expected 'module.path.Name'")
    if ".IFBench" in module_path:
        # IFBench's package import downloads a spaCy model through pip; it is locked instead.
        skip_spacy_model_download()
    module = import_module(module_path)
    obj = getattr(module, attr_name)

    if isinstance(obj, dspy.Module):
        return copy.deepcopy(obj)
    if isinstance(obj, type) and issubclass(obj, dspy.Module):
        kwargs = {}
        for name, param in inspect.signature(obj.__init__).parameters.items():
            if name == "self":
                continue
            if name in models:
                kwargs[name] = models[name]
            elif param.default is inspect.Parameter.empty:
                # Unmatched model roles use the primary model.
                warning(
                    f"dspy-program {dotted_path!r} param {name!r} not in models "
                    f"{sorted(models.keys())}; defaulting to 'model'."
                )
                kwargs[name] = models.get("model")
        return obj(**kwargs)
    raise TypeError(f"{dotted_path!r} is neither a dspy.Module subclass nor an instance.")


def create_dspy_program_uncompiled_graph(
    scenario: Agent,
    model: BaseChatModel,
    program_class: str = "",
    models: dict[str, BaseChatModel | Embeddings] | None = None,
    **kwargs: Any,
) -> StateGraph:
    """Return an uncompiled ``StateGraph`` that runs the given ``dspy.Module``."""
    import dspy  # pyright: ignore[reportMissingImports]

    from agents_gym_core.loops.dspy_lm import GymLM

    if not program_class:
        raise ValueError(
            "dspy-program loop requires a dotted path as the version, e.g.:\n"
            "  --loops dspy-program:gepa_artifact.benchmarks.hover.hover_program.HoverMultiHop"
        )

    model_name = get_model_name(model)
    lm = GymLM(model, model_name=model_name)

    # Build a models dict with GymLM wrappers for all models including primary.
    # _resolve_program matches constructor param names against this dict.
    lm_dict = {
        role: GymLM(m, model_name=get_model_name(m))
        for role, m in (models or {}).items()
        if isinstance(m, BaseChatModel)
    }

    program = _resolve_program(program_class, models=lm_dict)
    info(f"Resolved DSPy program: {program_class} ({type(program).__name__})")

    # Override predictor instructions from agent.yaml prompts.
    # Priority: signature_prompts > system_prompt > program default.
    for pred_name, predictor in program.named_predictors():
        short_name = pred_name.removesuffix(".predict")
        sig_prompt = scenario.signature_prompts.get(pred_name) or scenario.signature_prompts.get(
            short_name
        )
        if sig_prompt:
            predictor.signature = predictor.signature.with_instructions(sig_prompt)
        elif scenario.system_prompt:
            predictor.signature = predictor.signature.with_instructions(scenario.system_prompt)

        demos_raw = scenario.signature_demos.get(pred_name) or scenario.signature_demos.get(
            short_name
        )
        if demos_raw:
            input_fields = tuple(predictor.signature.input_fields.keys())
            predictor.demos = [dspy.Example(**d).with_inputs(*input_fields) for d in demos_raw]

    input_field_names = set(scenario.input_schema.model_fields.keys())
    output_field_names = set(scenario.output_schema.model_fields.keys())
    state_model = build_dspy_state_model(scenario)
    output_model = build_dspy_output_model(scenario)

    def dspy_program_node(state: Any) -> dict:
        state_dict = state if isinstance(state, dict) else dict(state)
        inputs = {k: state_dict[k] for k in input_field_names if k in state_dict}
        with dspy.context(lm=lm):
            prediction = program(**inputs)
        raw = {
            f: getattr(prediction, f)
            for f in output_field_names
            if getattr(prediction, f, None) is not None
        }
        missing = output_field_names - set(raw.keys())
        if missing:
            warning(f"DSPy program node: missing output fields {missing}")
        return output_model.model_validate(raw).model_dump(exclude_none=True)

    builder = StateGraph(state_schema=state_model, output_schema=output_model)
    builder.add_node("dspy_program", dspy_program_node)
    builder.add_edge(START, "dspy_program")
    builder.add_edge("dspy_program", END)

    models_info: dict[str, str] = {"model": model_name}
    for role, m in (models or {}).items():
        if role not in ("model", "embeddings") and isinstance(m, BaseChatModel):
            models_info[role] = get_model_name(m)

    builder.loop_details = LoopDetails(  # pyright: ignore[reportAttributeAccessIssue]
        loop_name=LoopName.DSPY_PROGRAM,
        loop_version=program_class,
        models=models_info,
    )
    return builder


def create_dspy_program_graph(
    scenario: Agent,
    model: BaseChatModel,
    program_class: str = "",
    models: dict[str, BaseChatModel | Embeddings] | None = None,
    checkpointer: Any | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build and compile a ``StateGraph`` that runs any ``dspy.Module``."""
    builder = create_dspy_program_uncompiled_graph(
        scenario, model, program_class=program_class, models=models, **kwargs
    )
    compiled = builder.compile(checkpointer=checkpointer)
    compiled.loop_details = builder.loop_details  # pyright: ignore[reportAttributeAccessIssue]
    return compiled


__all__ = [
    "create_dspy_program_graph",
    "create_dspy_program_uncompiled_graph",
]
