"""End-to-end DSPy-based GEPA optimization for a gym agent.

Connects Gym agents and their evaluators to upstream ``dspy.GEPA`` through:
- :class:`GymAgentModule` (one predictor whose instructions == system_prompt)
- :func:`make_gym_metric` (gym evaluator battery → ``dspy.Prediction(score, feedback)``)
- :func:`evalset_to_dspy_examples` (eval set → ``list[dspy.Example]``)

The adapter compiles the Gym student with
``dspy.GEPA(metric=..., reflection_lm=..., ...).compile(student, trainset, valset)``, then
returns the evolved system prompt and the optimized program.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import dspy
from langchain_core.language_models import BaseChatModel
from uipath.eval.evaluators import BaseEvaluator

from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.prompt_optimization.gepa_dspy.dataset import evalset_to_dspy_examples
from agents_gym_core.prompt_optimization.gepa_dspy.metric import make_gym_metric
from agents_gym_core.prompt_optimization.gepa_dspy.module import GymAgentModule
from agents_gym_core.utils.gym_types import Agent

logger = logging.getLogger(__name__)


@dataclass
class DSPyOptimizeConfig:
    auto: Literal["light", "medium", "heavy"] | None = None
    max_full_evals: int | None = None
    max_metric_calls: int | None = None
    reflection_minibatch_size: int = 5
    candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto"
    use_merge: bool = True
    num_threads: int = 8
    train_fraction: float = 0.8
    seed: int = 0
    loop_name: str = "basic"
    loop_version: str = DEFAULT_LOOP_VERSION


def optimize_dspy(
    *,
    agent: Agent,
    evaluators: dict[str, BaseEvaluator],
    model: BaseChatModel,
    reflection_lm: dspy.BaseLM,
    val_agent: Agent | None = None,
    config: DSPyOptimizeConfig | None = None,
    log_dir: Path | None = None,
) -> tuple[str, GymAgentModule]:
    """Run DSPy GEPA on ``agent``. Returns ``(evolved_system_prompt, best_module)``.

    If ``val_agent`` is provided, its evaluation_set is used as DSPy's valset
    (skipping the internal ``train_fraction`` split). Otherwise the trainset is
    internally shuffled and split per ``cfg.train_fraction``. With
    ``train_fraction=1.0``, as in ``gepa/benchmark.py``, the trainset also
    serves as the valset.
    """
    cfg = config or DSPyOptimizeConfig()

    if val_agent is not None:
        trainset = evalset_to_dspy_examples(agent)
        if not trainset:
            raise ValueError(f"Agent {agent.name!r} has no evaluation datapoints")
        valset = evalset_to_dspy_examples(val_agent)
        if not valset:
            raise ValueError(f"val_agent {val_agent.name!r} has no evaluation datapoints")
        logger.info(
            "optimize_dspy: agent=%s train=%d val=%d (caller-provided splits) evaluators=%d",
            agent.name,
            len(trainset),
            len(valset),
            len(evaluators),
        )
    else:
        examples = evalset_to_dspy_examples(agent)
        if not examples:
            raise ValueError(f"Agent {agent.name!r} has no evaluation datapoints")
        rng = random.Random(cfg.seed)
        rng.shuffle(examples)
        n_train = max(1, int(len(examples) * cfg.train_fraction))
        trainset = examples[:n_train]
        if n_train < len(examples):
            valset = examples[n_train:]
            logger.info(
                "optimize_dspy: agent=%s train=%d val=%d (internal %.0f/%.0f split) evaluators=%d",
                agent.name,
                len(trainset),
                len(valset),
                cfg.train_fraction * 100,
                (1 - cfg.train_fraction) * 100,
                len(evaluators),
            )
        else:
            # train_fraction=1.0: as in the paper, the train split is both DSPy's
            # trainset and its Pareto valset. The test split (e.g. ``<agent>:test``)
            # is never passed in; it is used only for the final comparison.
            valset = trainset
            logger.info(
                "optimize_dspy: agent=%s train=%d val=%d (val=train) evaluators=%d",
                agent.name,
                len(trainset),
                len(valset),
                len(evaluators),
            )

    student = GymAgentModule(
        agent,
        model=model,
        loop_name=cfg.loop_name,
        loop_version=cfg.loop_version,
    )

    metric = make_gym_metric(
        evaluators=evaluators,
        evaluator_weights=dict(agent.evaluator_weights or {}),
        llm_calls_dir=Path(log_dir) / "llm_calls" if log_dir else None,
    )

    # Exactly one budget knob must be set
    budget_kwargs: dict = {}
    if cfg.auto is not None:
        budget_kwargs["auto"] = cfg.auto
    elif cfg.max_metric_calls is not None:
        budget_kwargs["max_metric_calls"] = cfg.max_metric_calls
    elif cfg.max_full_evals is not None:
        budget_kwargs["max_full_evals"] = cfg.max_full_evals
    else:
        # Default: cap by full evals so it terminates predictably
        budget_kwargs["max_full_evals"] = 5

    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,  # pyright: ignore[reportArgumentType]  # any dspy.BaseLM works
        reflection_minibatch_size=cfg.reflection_minibatch_size,
        candidate_selection_strategy=cfg.candidate_selection_strategy,
        use_merge=cfg.use_merge,
        num_threads=cfg.num_threads,
        log_dir=str(log_dir) if log_dir else None,
        track_stats=True,
        seed=cfg.seed,
        **budget_kwargs,
    )

    # Hook the reflection LM so every proposer call is dumped to JSONL for
    # post-hoc inspection (langsmith-style local trace). dspy.context scopes the callback to
    # this optimization: dspy.settings.configure may only be called from the thread that first
    # configured dspy, and agents optimized in parallel would share one callback list.
    callbacks = list(dspy.settings.callbacks or [])
    if log_dir is not None:
        from agents_gym_core.prompt_optimization.gepa.io_logger import ReflectionCallLogger

        callbacks.append(ReflectionCallLogger(Path(log_dir) / "llm_calls" / "reflection.jsonl"))

    with dspy.context(callbacks=callbacks):
        optimized = optimizer.compile(student, trainset=trainset, valset=valset)

    # The optimized program is a GymAgentModule with the evolved instructions
    # in its predictor.
    evolved = optimized.predict.signature.instructions  # type: ignore[attr-defined]
    return str(evolved), cast(GymAgentModule, optimized)


def load_agent_and_evaluators(agent_name: str, agent_version: str = "base"):
    """Convenience: load agent + evaluators via CoreAgentRegistry."""
    from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry

    registry = CoreAgentRegistry()
    agent = registry.get_agent(agent_name, agent_version)
    evaluators = EvaluationLoader.get_evaluators(
        agent_name=agent.name,
        directory_provider=registry.get_directory_provider(),
    )
    return registry, agent, evaluators
