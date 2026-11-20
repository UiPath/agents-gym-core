"""End-to-end GEPA benchmark on the paper's agents.

The paper's GEPA-DSPy settings are the ones ``scripts/run_gepa.sh`` passes.

For each agent:
  1. (Optional) ``--reset`` first overwrites ``prompts/system_prompt.md``
     and ``prompts/user_prompt.md`` with their ``prompts/baseline/`` baselines
     so optimization starts from the unoptimized baseline. Mirrors the
     ``react_optimizer`` --reset behaviour.
  2. Prepare or reuse the agent's ``train`` / ``test`` splits.
  3. Load ``<agent>:train`` and optimize the seed system_prompt with the native optimizer
     (:func:`agents_gym_core.prompt_optimization.gepa.optimize.optimize`) or, with
     ``--variant dspy`` (the paper's enterprise baseline), with ``dspy.GEPA``
     (:func:`agents_gym_core.prompt_optimization.gepa_dspy.optimize.optimize_dspy`). ``train`` uses the agent's current
     ``system_prompt.md``: the unoptimized baseline with ``--reset``, otherwise
     the agent's default prompt.
  4. Evaluate **both** the seed and the evolved candidate on the held-out
     ``test`` datapoints (same agent, tools, and evaluators).
  5. Optionally write the evolved prompt as ``<agent>:gepa_<runid>`` (``gepa_dspy_<runid>``
     for the dspy variant; ``<runid>`` is the run's ``YYYYmmdd_HHMMSS``) so it can be re-run
     via ``gym eval``.

Final output: a Markdown comparison table + a JSON dump under ``run_dir/``.

Usage::

    python -m agents_gym_core.prompt_optimization.gepa.benchmark
    python -m agents_gym_core.prompt_optimization.gepa.benchmark sql meeting_management
    python -m agents_gym_core.prompt_optimization.gepa.benchmark --reset --variant dspy --dspy-auto light
"""

import argparse
import asyncio
import importlib.util
import inspect
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader
from agents_gym_core.models.backend import configure_model_resolution
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.prompt_optimization.gepa.optimize import OptimizeConfig, optimize
from agents_gym_core.prompt_optimization.gepa.rollout import SYSTEM_PROMPT_KEY, rollout_batch
from agents_gym_core.prompt_optimization.gepa.writeback import save_evolved_version
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider

if TYPE_CHECKING:
    import dspy

# Imported lazily inside the dspy branch so the native variant doesn't
# pay the dspy import cost.

logger = logging.getLogger(__name__)

DEFAULT_AGENTS = [
    "sql",
    "customer_support_triage_agent",
    "meeting_management",
    "product_questions_agent",
    "claim_history",
]


@dataclass
class BenchmarkRow:
    agent: str
    seed_train: float
    seed_val: float | None
    seed_test: float
    best_train: float
    best_val: float | None
    best_test: float
    n_candidates: int
    seed_chars: int
    evolved_chars: int
    version_name: str | None
    duration_s: float
    variant: str = "native"
    error: str | None = None
    # True when GEPA kept the seed prompt; otherwise evolved_prompt_path is the prompt this
    # run wrote back (both set only on success, so callers never guess from the filesystem)
    seed_kept: bool = False
    evolved_prompt_path: str | None = None
    # Total optimizer metric calls (dspy variant only, from dspy.GEPA's detailed_results).
    metric_calls: int | None = None


def ensure_splits(agent_name: str) -> None:
    """Prepare an agent's ``_train`` / ``_test`` splits, reusing existing splits.

    Load ``init.py`` directly via ``spec_from_file_location`` when either split
    needs initialization.
    """
    agent_dir = CoreDirectoryProvider().get_agents_dir() / agent_name
    eval_dir = agent_dir / "evaluations" / "eval-sets"
    train = eval_dir / f"evaluation-set-{agent_name}_train.json"
    test = eval_dir / f"evaluation-set-{agent_name}_test.json"
    if train.exists() and test.exists():
        return
    init_path = agent_dir / "init.py"
    if not init_path.exists():
        raise FileNotFoundError(f"No init.py for {agent_name} at {init_path}")
    spec = importlib.util.spec_from_file_location(f"_gepa_init_{agent_name}", init_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {init_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init(agent_dir, {})


async def benchmark_one(
    agent_name: str,
    *,
    variant: str,
    registry: CoreAgentRegistry,
    model: BaseChatModel,
    proposer_lm: BaseChatModel,
    reflection_lm: "dspy.BaseLM | None",
    cfg: OptimizeConfig,
    dspy_overrides: dict,
    log_dir: Path,
    write_back: bool,
    reset: bool = False,
) -> BenchmarkRow:
    started = time.time()
    try:
        if reset:
            from agents_gym_core.prompt_optimization.reset_prompts import reset_prompts

            logger.info("--reset: restoring %s prompts to original baseline", agent_name)
            reset_prompts(agent_name)
        ensure_splits(agent_name)
        train_agent = registry.get_agent(agent_name, "train")
        test_agent = registry.get_agent(agent_name, "test")
        evaluators = EvaluationLoader.get_evaluators(
            agent_name=agent_name,
            directory_provider=registry.get_directory_provider(),
        )
        if not evaluators:
            raise RuntimeError(f"No evaluators for {agent_name}")

        agent_log_dir = log_dir / agent_name

        seed_text = train_agent.system_prompt
        seed_chars = len(seed_text)

        best: dict[str, str]
        evolved_text: str
        metric_calls: int | None = None

        if variant == "native":
            best, state = await optimize(
                agent=train_agent,
                model=model,
                proposer_lm=proposer_lm,
                evaluators=evaluators,
                config=cfg,
                log_dir=agent_log_dir,
            )
            best_idx = state.best_idx()
            seed_train = state.train_mean(0)
            seed_val = state.val_scores[0]
            best_train = state.train_mean(best_idx)
            best_val = state.val_scores[best_idx]
            n_candidates = len(state.candidates)
            evolved_text = best[SYSTEM_PROMPT_KEY]
        elif variant == "dspy":
            from agents_gym_core.prompt_optimization.gepa_dspy.optimize import (
                DSPyOptimizeConfig,
                optimize_dspy,
            )

            # DSPy expects exactly one budget knob. Precedence:
            #   --dspy-auto > --dspy-max-metric-calls > --dspy-max-full-evals.
            auto = dspy_overrides.get("auto")
            max_metric_calls = dspy_overrides.get("max_metric_calls")
            # As in the paper, GEPA optimizes on the full ``train`` split, which
            # also serves as its validation set. ``test`` is never passed to the
            # optimizer; it is used only for the final seed-vs-evolved comparison.
            dspy_cfg = DSPyOptimizeConfig(
                auto=auto,
                max_metric_calls=None if auto else max_metric_calls,
                max_full_evals=None
                if (auto or max_metric_calls)
                else dspy_overrides["max_full_evals"],
                reflection_minibatch_size=cfg.minibatch_size,
                use_merge=dspy_overrides["use_merge"],
                num_threads=dspy_overrides["num_threads"],
                train_fraction=1.0,
                seed=cfg.seed,
                loop_name=cfg.loop_name,
                loop_version=cfg.loop_version,
            )
            if reflection_lm is None:
                raise ValueError("the dspy variant needs a reflection LM")
            evolved_text, optimized = await asyncio.to_thread(
                optimize_dspy,
                agent=train_agent,
                evaluators=evaluators,
                model=model,
                reflection_lm=reflection_lm,
                config=dspy_cfg,
                log_dir=agent_log_dir,
            )
            best = {SYSTEM_PROMPT_KEY: evolved_text}
            # dspy.GEPA keeps its minibatch (train) scores internal, so the train columns stay
            # empty. With track_stats=True (gepa_dspy/optimize.py) the program carries
            # `detailed_results`: every candidate, its aggregate score on dspy's valset (the
            # train split here, as train_fraction=1.0) and the total metric calls.
            seed_train = float("nan")
            best_train = float("nan")
            # dspy stores the seed as inspect.cleandoc(seed_text): a kept seed comes back so.
            kept_seed = evolved_text.strip() == inspect.cleandoc(seed_text).strip()
            best_idx = 0 if kept_seed else 1
            stats = getattr(optimized, "detailed_results", None)
            if stats is not None:
                n_candidates = len(stats.candidates)
                seed_val = stats.val_aggregate_scores[0]
                best_val = stats.val_aggregate_scores[stats.best_idx]
                metric_calls = stats.total_metric_calls
            else:
                n_candidates = 1 if kept_seed else 2
                seed_val = None
                best_val = None
        else:
            raise ValueError(f"Unknown variant {variant!r}")

        test_dps = list(test_agent.evaluation_set.evaluations or [])
        weights = dict(test_agent.evaluator_weights or {})

        # Dump test-eval rollouts to a separate file so they don't mix
        # with optimization-time rollouts in `rollouts.jsonl`.
        test_llm_calls_dir = agent_log_dir / "llm_calls" / "test_eval"

        seed_candidate: dict[str, str] = {SYSTEM_PROMPT_KEY: seed_text}
        seed_test_results = await rollout_batch(
            seed_candidate,
            test_agent,
            test_dps,
            model,
            evaluators,
            weights,
            max_concurrency=cfg.max_concurrency,
            loop_name=cfg.loop_name,
            loop_version=cfg.loop_version,
            llm_calls_dir=test_llm_calls_dir / "seed",
        )
        seed_test = sum(r.aggregated_score for r in seed_test_results) / len(seed_test_results)

        if best_idx == 0:
            best_test = seed_test
        else:
            best_test_results = await rollout_batch(
                best,
                test_agent,
                test_dps,
                model,
                evaluators,
                weights,
                max_concurrency=cfg.max_concurrency,
                loop_name=cfg.loop_name,
                loop_version=cfg.loop_version,
                llm_calls_dir=test_llm_calls_dir / "evolved",
            )
            best_test = sum(r.aggregated_score for r in best_test_results) / len(best_test_results)

        version_name: str | None = None
        evolved_prompt_path: str | None = None
        writeback_error: str | None = None
        if write_back and best_idx != 0:
            run_id = log_dir.name.removeprefix("benchmark_")  # date and time, as in main()
            version_name = f"gepa{'_dspy' if variant == 'dspy' else ''}_{run_id}"
            try:
                prompt_path, _ = save_evolved_version(
                    directory_provider=registry.get_directory_provider(),
                    agent_name=agent_name,
                    version_name=version_name,
                    evolved_system_prompt=evolved_text,
                    source_version="train",
                    user_prompt=train_agent.user_prompt,
                )
                evolved_prompt_path = str(prompt_path)
            except Exception as e:
                # Preserve test results and record the writeback error separately.
                logger.warning("writeback failed for %s: %s", agent_name, e)
                version_name = None
                writeback_error = f"writeback failed: {type(e).__name__}: {e}"

        return BenchmarkRow(
            agent=agent_name,
            seed_train=seed_train,
            seed_val=seed_val,
            seed_test=seed_test,
            best_train=best_train,
            best_val=best_val,
            best_test=best_test,
            n_candidates=n_candidates,
            seed_chars=seed_chars,
            evolved_chars=len(evolved_text),
            version_name=version_name,
            duration_s=time.time() - started,
            variant=variant,
            error=writeback_error,
            seed_kept=best_idx == 0,
            evolved_prompt_path=evolved_prompt_path,
            metric_calls=metric_calls,
        )
    except Exception as e:
        logger.exception("benchmark_one failed for %s", agent_name)
        return BenchmarkRow(
            agent=agent_name,
            seed_train=float("nan"),
            seed_val=None,
            seed_test=float("nan"),
            best_train=float("nan"),
            best_val=None,
            best_test=float("nan"),
            n_candidates=0,
            seed_chars=0,
            evolved_chars=0,
            version_name=None,
            duration_s=time.time() - started,
            variant=variant,
            error=f"{type(e).__name__}: {e}",
        )


def render_markdown_table(rows: list[BenchmarkRow]) -> str:
    header = (
        "| Agent | Seed train | Seed val | Seed test | Evolved train | Evolved val | "
        "Evolved test | Δ test | Cands | Chars Δ | Version | Time |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
    )
    lines = [header]
    for r in rows:
        if r.error:
            lines.append(
                f"| {r.agent} | — | — | — | — | — | — | — | — | — | ERROR: `{r.error}` | {r.duration_s:.0f}s |"
            )
            continue

        def f(x: float | None) -> str:
            return "—" if x is None else f"{x:.3f}"

        delta = r.best_test - r.seed_test
        sign = "+" if delta >= 0 else ""
        chars_delta = f"{r.seed_chars}→{r.evolved_chars}"
        version = r.version_name or "(seed)"
        lines.append(
            f"| {r.agent} | {f(r.seed_train)} | {f(r.seed_val)} | {f(r.seed_test)} | "
            f"{f(r.best_train)} | {f(r.best_val)} | {f(r.best_test)} | "
            f"{sign}{delta:.3f} | {r.n_candidates} | {chars_delta} | "
            f"{version} | {r.duration_s:.0f}s |"
        )
    return "\n".join(lines)


def _model_by_name(name: str) -> BaseChatModel:
    """Resolve a model on the backend that ``AGENTS_GYM_LLM_BACKEND`` selects."""
    return get_chat_model(name)


async def _amain(args: argparse.Namespace) -> None:
    configure_model_resolution()
    registry = CoreAgentRegistry()
    model = _model_by_name(args.model)
    proposer = _model_by_name(args.proposer_model)
    reflection_lm = None
    if args.variant == "dspy":
        from agents_gym_core.prompt_optimization.gepa_dspy.lm import make_dspy_lm

        reflection_lm = make_dspy_lm(
            args.proposer_model,
            max_tokens=args.reflection_max_tokens,
            temperature=args.reflection_temperature,
        )

    cfg = OptimizeConfig(
        num_iters=args.num_iters,
        minibatch_size=args.minibatch,
        train_fraction=args.train_fraction,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
    )

    dspy_overrides = {
        "auto": args.dspy_auto,
        "max_full_evals": args.dspy_max_full_evals,
        "max_metric_calls": args.dspy_max_metric_calls,
        "use_merge": not args.dspy_no_merge,
        "num_threads": args.dspy_num_threads
        if args.dspy_num_threads is not None
        else args.max_concurrency,
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir / f"benchmark_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_dir = %s", run_dir)

    sem = asyncio.Semaphore(args.parallel_agents)

    async def _run(name: str) -> BenchmarkRow:
        async with sem:
            logger.info("--- starting %s ---", name)
            one = benchmark_one(
                name,
                variant=args.variant,
                registry=registry,
                model=model,
                proposer_lm=proposer,
                reflection_lm=reflection_lm,
                cfg=cfg,
                dspy_overrides=dspy_overrides,
                log_dir=run_dir,
                write_back=not args.no_writeback,
                reset=args.reset,
            )
            if args.variant == "dspy":
                # GEPA-DSPy makes its model calls on its own event loop. The agent's benchmark
                # runs there too, so the final test evaluation uses the model client on the
                # loop that opened its connections.
                from agents_gym_core.prompt_optimization.gepa_dspy.module import EVENT_LOOP

                one = asyncio.wrap_future(asyncio.run_coroutine_threadsafe(one, EVENT_LOOP))
            row = await one
            logger.info(
                "--- done %s in %.0fs (best_test=%.3f, Δ=%+.3f) ---",
                name,
                row.duration_s,
                row.best_test,
                row.best_test - row.seed_test if row.error is None else 0.0,
            )
            return row

    rows: list[BenchmarkRow] = []
    tasks = [asyncio.create_task(_run(name)) for name in args.agents]
    for finished in asyncio.as_completed(tasks):
        row = await finished
        rows.append(row)
        # Persist after each agent so a crash mid-benchmark doesn't lose results
        (run_dir / "rows.json").write_text(
            json.dumps([asdict(r) for r in rows], indent=2),
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text(
            render_markdown_table(rows) + "\n",
            encoding="utf-8",
        )

    print()
    print(render_markdown_table(rows))
    print()
    print(f"Results: {run_dir}")
    failed = [r.agent for r in rows if r.error]
    if failed:
        raise SystemExit(f"benchmark failed for: {', '.join(failed)} (see {run_dir / 'rows.json'})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents_gym_core.prompt_optimization.gepa.benchmark")
    parser.add_argument(
        "agents",
        nargs="*",
        default=DEFAULT_AGENTS,
        help=f"Agent names (default: {' '.join(DEFAULT_AGENTS)})",
    )
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--proposer-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Concurrent rollouts per agent (per-agent budget)",
    )
    parser.add_argument(
        "--parallel-agents",
        type=int,
        default=5,
        help="Number of agents optimized concurrently (default: 5)",
    )
    parser.add_argument(
        "--variant",
        choices=["native", "dspy"],
        default="native",
        help="GEPA implementation: gym-native (default) or upstream dspy.GEPA",
    )
    parser.add_argument(
        "--dspy-max-full-evals",
        type=int,
        default=5,
        help="Budget for dspy.GEPA in 'full evals'-equivalents "
        "(ignored if --dspy-auto or --dspy-max-metric-calls is set)",
    )
    parser.add_argument(
        "--dspy-max-metric-calls",
        type=int,
        default=None,
        help="Budget for dspy.GEPA as an absolute metric-call count "
        "(overrides --dspy-max-full-evals; ignored if --dspy-auto is set)",
    )
    parser.add_argument(
        "--dspy-auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="dspy.GEPA preset budget (overrides --dspy-max-full-evals "
        "and --dspy-max-metric-calls)",
    )
    parser.add_argument(
        "--dspy-no-merge",
        action="store_true",
        help="Disable dspy.GEPA program merging (use_merge=False)",
    )
    parser.add_argument(
        "--dspy-num-threads",
        type=int,
        default=None,
        help="dspy.GEPA num_threads (defaults to --max-concurrency)",
    )
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=0.0,
        help="Temperature for the dspy reflection LM (gpt-5 reasoning models only accept 1.0)",
    )
    parser.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=32000,
        help="max_tokens for the DSPy reflection LM",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/gepa_benchmark"))
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="Don't add a versions.yaml entry for the evolved prompt",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Before optimizing, copy each agent's prompts/baseline/*.md over "
        "prompts/*.md to start from the unoptimized baseline (matches the react "
        "optimizer's --reset behaviour). The mutation is in-place; the "
        "agent's default system_prompt.md will be overwritten.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
