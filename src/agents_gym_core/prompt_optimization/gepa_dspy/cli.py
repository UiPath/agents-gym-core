"""DSPy-GEPA CLI — same shape as ``agents_gym_core.prompt_optimization.gepa.cli`` for easy comparison.

The paper's GEPA-DSPy settings are the ones ``scripts/run_gepa.sh`` passes.

Example::

    python -m agents_gym_core.prompt_optimization.gepa_dspy sql --version baseline-train --max-full-evals 5
"""

import argparse
import inspect
import logging
import time
from pathlib import Path

from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.models.backend import configure_model_resolution
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.prompt_optimization.gepa.writeback import save_evolved_version
from agents_gym_core.prompt_optimization.gepa_dspy.lm import make_dspy_lm
from agents_gym_core.prompt_optimization.gepa_dspy.optimize import (
    DSPyOptimizeConfig,
    load_agent_and_evaluators,
    optimize_dspy,
)
from agents_gym_core.reproduce import versions_of


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents_gym_core.prompt_optimization.gepa_dspy")
    parser.add_argument("agent")
    parser.add_argument(
        "--version", help="Agent version (default: train if the agent has it, else base)"
    )
    parser.add_argument("--loop-name", default="basic")
    parser.add_argument("--loop-version", default=DEFAULT_LOOP_VERSION)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--reflection-model", default="gpt-4o-2024-11-20")
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=0.0,
        help="Reflection LM temperature (gpt-5 reasoning models only accept 1.0; the paper used 1.0)",
    )
    parser.add_argument(
        "--reflection-max-tokens", type=int, default=16000, help="Reflection LM max_tokens"
    )
    # Budget — exactly one
    parser.add_argument("--auto", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-full-evals", type=int)
    parser.add_argument("--max-metric-calls", type=int)
    # Other knobs
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument(
        "--candidate-selection",
        choices=["pareto", "current_best"],
        default="pareto",
    )
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/gepa_dspy"))
    parser.add_argument("--no-writeback", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.version = args.version or ("train" if "train" in versions_of(args.agent) else "base")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    configure_model_resolution()

    registry, agent, evaluators = load_agent_and_evaluators(args.agent, args.version)
    if not evaluators:
        raise SystemExit(f"No evaluators for {args.agent!r}")

    model = get_chat_model(args.model)  # on the configured backend

    reflection_lm = make_dspy_lm(
        args.reflection_model,
        max_tokens=args.reflection_max_tokens,
        temperature=args.reflection_temperature,
    )

    cfg = DSPyOptimizeConfig(
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.minibatch,
        candidate_selection_strategy=args.candidate_selection,
        use_merge=not args.no_merge,
        num_threads=args.num_threads,
        train_fraction=args.train_fraction,
        seed=args.seed,
        loop_name=args.loop_name,
        loop_version=args.loop_version,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = args.run_dir / f"{args.agent}_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    evolved, _optimized = optimize_dspy(
        agent=agent,
        evaluators=evaluators,
        model=model,
        reflection_lm=reflection_lm,
        config=cfg,
        log_dir=log_dir,
    )

    print()
    print("=" * 60)
    print(f"DSPy-GEPA done. Evolved prompt length: {len(evolved)} chars")
    print(f"Logs:           {log_dir}")
    print("=" * 60)

    # dspy stores the seed as inspect.cleandoc(seed), so that is what a kept seed comes back as.
    if not args.no_writeback and evolved.strip() != inspect.cleandoc(agent.system_prompt).strip():
        version_name = f"gepa_dspy_{run_id}"
        prompt_path, versions_yaml = save_evolved_version(
            directory_provider=registry.get_directory_provider(),
            agent_name=args.agent,
            version_name=version_name,
            evolved_system_prompt=evolved,
            source_version=args.version,
            user_prompt=agent.user_prompt,
        )
        print()
        print(f"Wrote prompt:   {prompt_path}")
        print(f"Updated:        {versions_yaml}")
        print(f"New version:    {args.agent}:{version_name}")


if __name__ == "__main__":
    main()
