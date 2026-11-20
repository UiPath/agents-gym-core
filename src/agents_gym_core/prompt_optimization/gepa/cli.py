"""GEPA CLI: ``python -m agents_gym_core.prompt_optimization.gepa <agent> [options]``.

Example::

    python -m agents_gym_core.prompt_optimization.gepa calculator --num-iters 5 --minibatch 3
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.models.backend import configure_model_resolution
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.prompt_optimization.gepa.optimize import OptimizeConfig, optimize
from agents_gym_core.prompt_optimization.gepa.rollout import SYSTEM_PROMPT_KEY
from agents_gym_core.prompt_optimization.gepa.writeback import save_evolved_version
from agents_gym_core.reproduce import versions_of


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents_gym_core.prompt_optimization.gepa")
    parser.add_argument("agent", help="Agent name (e.g. calculator)")
    parser.add_argument(
        "--version", help="Agent version (default: train if the agent has it, else base)"
    )
    parser.add_argument("--loop-name", default="basic")
    parser.add_argument("--loop-version", default=DEFAULT_LOOP_VERSION)
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-11-20",
        help="LLM driving the agent under test",
    )
    parser.add_argument(
        "--proposer-model",
        default="gpt-4o-2024-11-20",
        help="LLM for the GEPA instruction proposer",
    )
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiments/gepa"),
        help="Where iteration logs are written",
    )
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Don't write the evolved prompt as a new versions.yaml entry",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.version = args.version or ("train" if "train" in versions_of(args.agent) else "base")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    configure_model_resolution()

    registry = CoreAgentRegistry()
    agent = registry.get_agent(args.agent, args.version)
    evaluators = EvaluationLoader.get_evaluators(
        agent_name=agent.name,
        directory_provider=registry.get_directory_provider(),
    )
    if not evaluators:
        raise SystemExit(f"No evaluators found for agent {args.agent!r}")

    model = _model_by_name(args.model)
    proposer = _model_by_name(args.proposer_model)

    cfg = OptimizeConfig(
        num_iters=args.num_iters,
        minibatch_size=args.minibatch,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        loop_name=args.loop_name,
        loop_version=args.loop_version,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = args.run_dir / f"{args.agent}_{run_id}"

    best, state = asyncio.run(
        optimize(
            agent=agent,
            model=model,
            proposer_lm=proposer,
            evaluators=evaluators,
            config=cfg,
            log_dir=log_dir,
        )
    )

    best_idx = state.best_idx()
    print()
    print("=" * 60)
    print(f"GEPA done. Best candidate idx={best_idx} of {len(state.candidates)} total")
    print(f"  Train mean: {state.train_mean(best_idx):.3f}")
    val = state.val_scores[best_idx]
    print(f"  Val mean:   {val:.3f}" if val is not None else "  Val mean:   (no val split)")
    print(f"  Logs:       {log_dir}/iterations.jsonl")
    print("=" * 60)

    if args.no_writeback or best_idx == 0:
        if best_idx == 0:
            print("(seed candidate is still best — nothing to write back)")
        return

    version_name = f"gepa_{run_id}"
    prompt_path, versions_yaml = save_evolved_version(
        directory_provider=registry.get_directory_provider(),
        agent_name=args.agent,
        version_name=version_name,
        evolved_system_prompt=best[SYSTEM_PROMPT_KEY],
        source_version=args.version,
        user_prompt=agent.user_prompt,
    )
    print()
    print(f"Wrote prompt:   {prompt_path}")
    print(f"Updated:        {versions_yaml}")
    print(f"New version:    {args.agent}:{version_name}")
    print("Evaluate via:")
    print(f"  gym eval --agents {args.agent}:{version_name}")


def _model_by_name(name: str) -> BaseChatModel:
    """Resolve a model on the backend that ``AGENTS_GYM_LLM_BACKEND`` selects."""
    return get_chat_model(name)


if __name__ == "__main__":
    main()
