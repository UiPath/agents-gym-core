"""CLI for ReAPO prompt optimization (react and claude engines).

Usage:
    uv run gym optimize <agent> [options]

Examples:
    uv run gym optimize calculator
    uv run gym optimize calculator --engine react
    uv run gym optimize calculator --engine claude -r 6
    uv run gym optimize calculator --skill list
    uv run gym optimize calculator --continue
    uv run gym optimize claim_history --reset  # from the baseline, as in the paper; optimizes :train
"""

import argparse
import asyncio
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import redirect_stdout, suppress
from io import TextIOWrapper
from pathlib import Path

import yaml

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.models.backend import set_default_backend
from agents_gym_core.models.overrides import apply_configured_model_overrides
from agents_gym_core.models.utils import ModelBackend
from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.best_round import (
    LAST_PROMPTS_DIR,
    report_best_round,
    restore_best_round,
    restore_last_prompts,
    save_last_prompts,
)
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    EVAL_LOOPS_ENV,
    EVAL_MODEL_ENV,
    TARGET_ENV,
    compute_output_dir,
    directory_provider,
    init_started_experiments,
    load_json_safe,
    optimization_context,
    parse_spec,
    repo_root,
    require_agent_dir,
    target_eval_sets,
)
from agents_gym_core.prompt_optimization.mcp_tools.tools_runner import loop_specs
from agents_gym_core.prompt_optimization.react_optimizer.optimizer import (
    ABLATION_REMOVED_TOOLS,
    run_react_optimizer,
)
from agents_gym_core.prompt_optimization.react_optimizer.prompts import DEFAULT_SKILL, skill_names
from agents_gym_core.prompt_optimization.run_accounting import sessions
from agents_gym_core.utils.gym_settings_core import load_env_file

# Aliases for Claude models — maps short names to LLM gateway model names.
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "opus": "anthropic.claude-opus-4-6-v1",
    "sonnet": "anthropic.claude-sonnet-4-6",
}
STREAM_PARSER = Path(__file__).resolve().parent / "claude_optimizer" / "stream_parser.py"
MCP_SERVER = Path(__file__).resolve().parent / "mcp_server.py"

DEFAULT_MODEL = "gpt-5-2025-08-07"

# Version keys that point the agent at other prompt files than the ones the optimizer edits.
PROMPT_OVERRIDES = ("system_prompt", "user_prompt", "signature_prompts", "signature_demos")


def _datapoint_names(eval_sets: set[Path]) -> set[str]:
    names = set()
    for path in eval_sets:
        data, err = load_json_safe(path)
        items = data.get("evaluations") if not err and isinstance(data, dict) else None
        names |= {item["name"] for item in items or [] if isinstance(item, dict) and "name" in item}
    return names


def unsuitable_version(agent_spec: str) -> str | None:
    """Why ``gym optimize`` must not run on ``agent_spec``, or None.

    The optimizer edits the prompt files agent.yaml names, so a version that sets its own
    prompts would be evaluated on files it never edits; and it must not evaluate (and so read)
    the test split, which the agent's ``test`` version names.
    """
    agent, version = parse_spec(agent_spec)
    try:  # an unknown agent or version, or missing init data, stops the run before any model call
        CoreAgentRegistry(directory_provider=directory_provider()).get_agent(
            agent, version, tools=[]
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    agent_dir, err = require_agent_dir(agent, "all")
    if err or agent_dir is None:
        return err or f"Agent '{agent}' not found."
    try:
        versions = yaml.safe_load((agent_dir / "versions.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        versions = {}
    if not isinstance(versions, dict):
        versions = {}
    entry = versions.get(version) if version != "base" else None
    settings = (entry.get("agent") or {}) if isinstance(entry, dict) else {}
    if overrides := [key for key in PROMPT_OVERRIDES if key in settings]:
        return (
            f"{agent_spec} sets its own {', '.join(overrides)}, but the optimizer edits the "
            f"prompt files of {agent}/agent.yaml. Optimize a version that only selects eval "
            f"sets, such as {agent}:train."
        )
    if "test" in versions and version != "test":
        test = _datapoint_names(target_eval_sets(agent_dir, "test"))
        if shared := test & _datapoint_names(target_eval_sets(agent_dir, version)):
            return (
                f"{agent_spec} evaluates on {len(shared)} datapoints of {agent}'s test split. "
                f"Optimize {agent}:train."
            )
    elif version == "test":
        return f"{agent_spec} is the test split. Optimize {agent}:train."
    return None


# ---------------------------------------------------------------------------
# Shared: output dir + resume logic
# ---------------------------------------------------------------------------


def detect_last_round(output_dir: Path) -> int:
    """Return the highest existing round number, or -1 if none."""
    if not output_dir.is_dir():
        return -1
    max_round = -1
    for d in output_dir.iterdir():
        if d.is_dir():
            m = re.match(r"^round_(\d+)$", d.name)
            if m:
                max_round = max(max_round, int(m.group(1)))
    return max_round


def detect_last_completed_react_round(output_dir: Path) -> int:
    """Return the highest React round with a score from a full evaluation."""
    if not output_dir.is_dir():
        return -1
    completed_rounds = []
    for round_dir in output_dir.iterdir():
        if not round_dir.is_dir() or not (match := re.fullmatch(r"round_(\d+)", round_dir.name)):
            continue
        stats, err = load_json_safe(round_dir / "round_stats.json")
        score = stats.get("score") if not err and isinstance(stats, dict) else None
        # Partial evaluations also write round_stats.json, but do not supply a score.
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            completed_rounds.append(int(match.group(1)))
    return max(completed_rounds, default=-1)


def resolve_resume(output_dir: Path, mode: str) -> tuple[int, str]:
    """Resolve start_round and final mode given existing rounds and requested mode.

    Returns (start_round, mode) where mode is 'continue' or 'overwrite'.
    """
    last_round = detect_last_round(output_dir)
    # An existing workspace without rounds (a run stopped before round 0) still asks.
    if last_round < 0 and (mode or not output_dir.exists()):
        if mode == "overwrite" and output_dir.exists():  # e.g. a failed run's sessions
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        return 0, "continue" if mode == "continue" else "overwrite"

    if not mode:
        print(f"Existing workspace found: {output_dir.relative_to(repo_root())}/")
        if last_round >= 0:
            print(f"  Rounds 0..{last_round} already exist.")
        print()
        print(f"  [c] Continue from round {detect_last_completed_react_round(output_dir) + 1}")
        print("  [o] Overwrite (start fresh from round 0)")
        print()
        choice = input("Choose [c/o]: ").strip().lower()
        if choice == "c":
            mode = "continue"
        elif choice == "o":
            mode = "overwrite"
        else:
            print("Aborted.")
            sys.exit(1)

    if mode == "continue":
        # After the last scored round: an interrupted round's folder holds no score yet.
        return detect_last_completed_react_round(output_dir) + 1, "continue"
    else:
        print(f"Overwriting: removing entire output directory {output_dir.name}/")
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        return 0, "overwrite"


def recorded_loops(output_path: Path) -> str | None:
    """The eval loops the workspace's sessions recorded (its first that did), or None."""
    for session in sessions(output_path):
        if isinstance(session.get("loops"), str):
            return session["loops"]
    return None


def recorded_rounds(output_path: Path) -> int | None:
    """The number of rounds the workspace's run was started with (its first session), or None."""
    for session in sessions(output_path):
        if isinstance(session.get("num_rounds"), int):
            return session["num_rounds"]
    return None


def keep_recorded_rounds(args: argparse.Namespace, output_path: Path) -> None:
    """A continuation keeps the run's own number of rounds (IFBench runs 6)."""
    if (rounds := recorded_rounds(output_path)) is not None and rounds != args.rounds:
        print(f"-c keeps the run's {rounds} rounds (not -r {args.rounds})")
        args.rounds = rounds


def list_skills() -> None:
    print("Available skills:")
    for name in skill_names():
        print(f"  {name}")


# ---------------------------------------------------------------------------
# Claude engine
# ---------------------------------------------------------------------------


def write_mcp_config(output_path: Path) -> Path:
    """Register this checkout's coach-light server for the Claude CLI.

    Passed with ``--strict-mcp-config`` so the session never picks up a coach-light
    registered elsewhere on the machine (e.g. one from another checkout).
    The server inherits the Claude process environment (COACH_*, model credentials).
    """
    command = [sys.executable, str(MCP_SERVER)]
    provider = directory_provider()
    if isinstance(provider, ProjectDirectoryProvider):
        command += ["--project", str(provider.get_project_root())]
    config = {
        "mcpServers": {"coach-light": {"type": "stdio", "command": command[0], "args": command[1:]}}
    }
    path = output_path / "mcp_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def _claude_cli_cost(result: dict) -> dict:
    """The optimizer's usage from the Claude CLI's final message, at the CLI's (list) prices.

    The CLI reports tokens and cost per model but no per-model call counts, so the run's calls
    are its turns.
    """
    from agents_gym_core.runtime.cost_tracking import CostTotals, ModelUsage

    total = CostTotals(num_llm_calls=int(result.get("num_turns") or 0))
    for model, usage in (result.get("modelUsage") or {}).items():
        cached = int(usage.get("cacheReadInputTokens") or 0)
        model_usage = ModelUsage(
            # cache reads and writes are input tokens too, as in the other engines' records
            input_tokens=int(usage.get("inputTokens") or 0)
            + cached
            + int(usage.get("cacheCreationInputTokens") or 0),
            output_tokens=int(usage.get("outputTokens") or 0),
            cached_tokens=cached,
            cost_usd=float(usage.get("costUSD") or 0.0),
        )
        total.by_model[model] = model_usage
        total.input_tokens += model_usage.input_tokens
        total.output_tokens += model_usage.output_tokens
        total.cached_tokens += model_usage.cached_tokens
    total.total_cost_usd = float(result.get("total_cost_usd") or 0.0)
    return total.model_dump()


def run_claude(args: argparse.Namespace, output_dir: str, start_round: int) -> None:
    from datetime import UTC, datetime

    from agents_gym_core.prompt_optimization.react_optimizer.prompts import _load_skill
    from agents_gym_core.prompt_optimization.run_accounting import (
        SESSIONS_DIRNAME,
        print_run_summary,
        run_totals,
    )

    output_path = repo_root() / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    session_path = output_path / SESSIONS_DIRNAME / f"{started:%Y%m%dT%H%M%S_%fZ}.json"
    metadata = {
        "engine": "claude",
        "started_at": started.isoformat(),
        "agent_name": args.agent_spec,
        "model": args.model,
        "optimizer_model": args.optimizer_model,
        "num_rounds": args.rounds,
        "skill_version": args.skill,
        "budget_usd": args.budget,
        "loops": args.loops,
        "start_round": start_round,
        "status": "running",
    }
    # Recorded now, so -c finds the run's rounds and loops even if this session is interrupted.
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(metadata, indent=2) + "\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"claude_session_{timestamp}.log"
    raw_log = output_path / f"raw_stream_{timestamp}.jsonl"
    stderr_log = output_path / f"stderr_{timestamp}.log"

    # Load and substitute skill
    content = _load_skill(args.skill)
    content = content.replace("$ARGUMENTS[0]", args.agent_spec)
    content = content.replace("$ARGUMENTS[1]", args.model)
    content = content.replace("$ARGUMENTS[2]", str(args.rounds))
    content = content.replace("$ARGUMENTS[3]", output_dir)
    content = content.replace("$ARGUMENTS[4]", args.loops)

    append_prompt = (
        f"You are a prompt optimization agent. Do NOT ask the user questions — "
        f"just execute the optimization protocol autonomously. "
        f"Work through all {args.rounds} rounds without stopping."
    )
    if start_round > 0:
        append_prompt += (
            f" IMPORTANT: This is a continuation session. Start from round {start_round} "
            f"(rounds 0..{start_round - 1} already exist in the output directory). "
            f"Read the previous round scores and prompts to understand where you left off before proceeding."
        )

    print(f"{'=' * 40}")
    print("  ReAPO (claude engine)")
    print(f"  Agent:      {args.agent_spec}")
    print(f"  Eval model: {args.model}")
    print(f"  Optimizer:  {args.optimizer_model}")
    print(f"  Rounds:     {args.rounds} (start: {start_round})")
    print(f"  Skill:      {args.skill}")
    print(f"  Budget:     ${args.budget}")
    print(f"  Output:     {output_dir}/")
    print(f"{'=' * 40}\n")

    mcp_config = write_mcp_config(output_path)
    cmd = [
        "claude",
        "--print",
        "--mcp-config",
        str(mcp_config),
        "--strict-mcp-config",
        "--verbose",
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--model",
        args.optimizer_model,
        "--max-budget-usd",
        str(args.budget),
        "--tools",
        "",
        "--disallowedTools",
        "Bash Read Write Edit Grep Glob WebFetch WebSearch NotebookEdit Agent",
        "--allowedTools",
        (
            "mcp__coach-light__summarize_agent mcp__coach-light__inspect_agent "
            "mcp__coach-light__get_agent_files mcp__coach-light__read_agent_file "
            "mcp__coach-light__inspect_tools mcp__coach-light__get_prompt "
            "mcp__coach-light__update_prompt mcp__coach-light__run_agent_async "
            "mcp__coach-light__wait_for_run mcp__coach-light__get_run_status "
            "mcp__coach-light__get_experiment_results mcp__coach-light__get_full_results "
            "mcp__coach-light__get_agent_messages mcp__coach-light__get_datapoint "
            "mcp__coach-light__check_evals mcp__coach-light__list_experiments "
            "mcp__coach-light__save_optimizer_log mcp__coach-light__read_optimizer_log "
            "mcp__coach-light__get_usage_stats mcp__coach-light__reset_usage_stats"
        ),
        "--append-system-prompt",
        append_prompt,
        "-p",
        content,
    ]

    env = os.environ.copy()
    env["COACH_OUTPUT_DIR"] = output_dir
    env["COACH_MAX_CHARS"] = str(args.max_chars)
    env["COACH_IMMUTABLE_ROUNDS_BEFORE"] = str(start_round)  # as the react engine does
    # Remove API keys that would override Claude CLI's OAuth authentication
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_API_KEY", None)

    results: list[dict] = []  # the CLI's final "result" message
    with (
        open(stderr_log, "w") as stderr_fh,
        open(raw_log, "w") as raw_fh,
        open(log_file, "w") as log_fh,
    ):
        claude_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_fh,
            text=True,
            cwd=str(repo_root()),
            env=env,
        )
        parser_proc = subprocess.Popen(
            [sys.executable, "-u", str(STREAM_PARSER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(repo_root()),
        )
        assert claude_proc.stdout is not None
        assert parser_proc.stdin is not None
        assert parser_proc.stdout is not None

        def feed_parser(
            src: subprocess.Popen[str],
            dst_stdin: TextIOWrapper,
            raw: TextIOWrapper,
        ) -> None:
            """Tee claude stdout → raw_log + parser stdin concurrently."""
            assert src.stdout is not None
            for line in src.stdout:
                raw.write(line)
                raw.flush()
                if '"type":"result"' in line.replace(" ", ""):
                    try:
                        results.append(json.loads(line))
                    except ValueError:
                        pass
                dst_stdin.write(line)
                dst_stdin.flush()
            dst_stdin.close()

        feeder = threading.Thread(
            target=feed_parser,
            args=(claude_proc, parser_proc.stdin, raw_fh),
            daemon=True,
        )
        feeder.start()

        # Print parser output to terminal + log in real time
        for line in parser_proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()

        feeder.join()
        claude_proc.wait()
        parser_proc.wait()

    result = results[-1] if results else {}
    failed = bool(claude_proc.returncode or not result or result.get("is_error"))
    finished = datetime.now(UTC)
    metadata |= {
        "status": "failed" if failed else "completed",
        "finished_at": finished.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 1),
        "optimizer_cost": _claude_cli_cost(result),
    }
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(metadata, indent=2) + "\n")
    totals = run_totals(output_path)
    (output_path / "run_metadata.json").write_text(
        json.dumps({**metadata, **totals}, indent=2) + "\n"
    )
    if failed:
        reason = result.get("result") or f"exit code {claude_proc.returncode}, see {stderr_log}"
        sys.exit(f"gym optimize: the claude CLI failed: {reason}")
    if "accounting_error" not in totals:
        print_run_summary(totals)
    print(f"\n=== Done. Results in {output_dir}/ ===")


# ---------------------------------------------------------------------------
# React engine
# ---------------------------------------------------------------------------


async def run_react(
    args: argparse.Namespace, output_dir: str, start_round: int = 0, resume: bool = False
) -> None:
    apply_configured_model_overrides()
    await run_react_optimizer(
        agent_name=args.agent_spec,
        model=args.model,
        optimizer_model=args.optimizer_model,
        num_rounds=args.rounds,
        output_dir=output_dir,
        name=args.name,
        loops=args.loops,
        temperature=args.temperature,
        recursion_limit=args.recursion_limit,
        max_chars=args.max_chars,
        skill_version=args.skill,
        ablation=args.ablation,
        start_round=start_round,
        resume=resume,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Optimize an agent's prompts with ReAPO, through its LangGraph agent (react, "
        "the default) or the Claude Code CLI (claude).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "agent_spec",
        help="Agent or train-split version to optimize (e.g. 'calculator' or 'claim_history:train')",
    )

    # Shared
    parser.add_argument(
        "--engine",
        choices=["react", "claude"],
        default="react",
        help="ReAPO engine: react (default, LangGraph agent) or claude (Claude Code CLI)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for eval runs (default: {DEFAULT_MODEL}; the paper uses "
        "gpt-4.1-mini-2025-04-14 on the public benchmarks)",
    )
    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        default=5,
        help="Number of optimization rounds (default: 5; the paper uses 6 on IFBench); -c keeps the "
        "interrupted run's value",
    )
    parser.add_argument(
        "-s",
        "--skill",
        default=DEFAULT_SKILL,
        choices=[*skill_names(), "list"],
        help="Skill prompt: full (default), trimmed (skill prompt ablation), tool-free "
        "(tools ablation), or 'list'",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Output directory label (default: claude_optimizer or react_optimizer)",
    )
    parser.add_argument(
        "--loops",
        default=None,
        help="Loop spec for eval runs (e.g. 'basic:any'); -c keeps the interrupted run's value",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20_000,
        help="Max chars for get_agent_messages (default: 20000)",
    )

    # Resume
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "-c",
        "--continue",
        dest="mode",
        action="store_const",
        const="continue",
        help="Continue after the last scored round, from the optimizer's last prompts",
    )
    resume.add_argument(
        "-o",
        "--overwrite",
        dest="mode",
        action="store_const",
        const="overwrite",
        help="Overwrite existing workspace",
    )

    # Shared optimizer options (used by multiple engines)
    shared_opt = parser.add_argument_group("Optimizer options (shared)")
    shared_opt.add_argument(
        "--optimizer-model",
        default=None,
        help="Model for the optimizer itself (default: gpt-5-2025-08-07; opus for the claude "
        "engine). The claude engine takes Claude model names; react also accepts those aliases.",
    )
    shared_opt.add_argument(
        "--model_backend",
        choices=[b.value for b in ModelBackend],
        default=None,
        help=(
            "Backend for every model role (optimizer, eval runs, judges): 'uipath' (LLM Gateway) or "
            "'langchain' (provider API keys). Default: $AGENTS_GYM_LLM_BACKEND, else langchain."
        ),
    )
    shared_opt.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Optimizer LLM temperature (react engine)",
    )

    # Claude-specific
    claude = parser.add_argument_group("Claude engine options")
    claude.add_argument(
        "-b",
        "--budget",
        type=float,
        default=50.0,
        help="Claude CLI model budget in USD (default: 50; --engine claude)",
    )

    # React-specific
    react = parser.add_argument_group("React engine options")
    react.add_argument(
        "--recursion-limit",
        type=int,
        default=500,
        help="Max LangGraph recursion steps (React only)",
    )
    react.add_argument(
        "--ablation",
        choices=list(ABLATION_REMOVED_TOOLS),
        default="full",
        help="Tools ablation: full (all tools, default), score-only (aggregate scores only) "
        "or macro (keeps get_full_results); the paper runs both with -s tool-free",
    )

    # Reset prompts
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset agent prompts to prompts/baseline/ before running optimization, as the paper "
        "does; without it a paper agent starts from its released ReAPO prompts (with -c, only "
        "when the run has nothing to continue)",
    )

    return parser


def _main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    load_env_file()

    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if args.model_backend:
        # Before any model is resolved; subprocesses (MCP server, testbed runs) inherit it.
        set_default_backend(args.model_backend)

    # Without a version, an agent with a train split is optimized on it, not on its test data.
    if ":" not in args.agent_spec and "train" in CoreAgentRegistry(
        directory_provider=directory_provider()
    ).list_agent_versions(args.agent_spec):
        args.agent_spec += ":train"
        print(f"Optimizing {args.agent_spec} (the agent's train split).")
    if problem := unsuitable_version(args.agent_spec):
        sys.exit(f"gym optimize: {problem}")

    # Handle --skill list
    if args.skill == "list":
        list_skills()
        sys.exit(0)

    if args.reset:
        from agents_gym_core.prompt_optimization.reset_prompts import reset_prompts

        # Check the baseline exists before -o deletes anything; the reset itself runs later.
        check_output = io.StringIO()
        with redirect_stdout(check_output):
            ok = reset_prompts(
                args.agent_spec.split(":")[0], dry_run=True, directory_provider=directory_provider()
            )
        if not ok:
            print(check_output.getvalue(), end="")
            sys.exit(1)

    # Default name per engine
    if args.name is None:
        args.name = {
            "claude": "claude_optimizer",
            "react": "react_optimizer",
        }[args.engine]

    if args.optimizer_model is None:
        args.optimizer_model = "opus" if args.engine == "claude" else "gpt-5-2025-08-07"
    # Resolve Claude model aliases for the react engine (LLM gateway model names)
    if args.engine == "react":
        args.optimizer_model = CLAUDE_MODEL_ALIASES.get(args.optimizer_model, args.optimizer_model)

    # Compute output dir
    output_dir = compute_output_dir(
        str(args.agent_spec),
        str(args.optimizer_model),
        str(args.model),
        str(args.name),
        skill=str(args.skill),
    )
    if args.engine == "react" and args.ablation != "full":
        output_dir += f"_ablation_{args.ablation}"
    output_path = repo_root() / output_dir

    # Resume logic
    continuing = args.mode == "continue"
    if args.engine == "react":
        if args.mode == "continue":
            keep_recorded_rounds(args, output_path)
            start_round = detect_last_completed_react_round(output_path) + 1
            if 0 < start_round <= args.rounds:
                print(f"React optimizer: continuing from round {start_round}")
        else:
            start_round = 0
        if output_path.exists() and args.mode is None:
            sys.stdout.flush()  # in a log file, keeps this message after the output above
            sys.exit(f"{output_dir}/ exists: pass -c to continue it or -o to start over.")
        if output_path.exists() and args.mode == "overwrite":
            print(f"React optimizer: removing existing workspace {output_dir}/")
            shutil.rmtree(output_path)
    else:
        mode = args.mode or ""
        start_round, mode = resolve_resume(output_path, mode)
        continuing = mode == "continue"
        if continuing:
            keep_recorded_rounds(args, output_path)

    # Every round is scored with the same loops, so a continuation keeps the run's.
    recorded = recorded_loops(output_path) if continuing else None
    if recorded is None:
        args.loops = args.loops or ""
    elif args.loops is not None and loop_specs(args.loops) != loop_specs(recorded):
        sys.exit(
            f"gym optimize: {output_dir}/ evaluates with --loops '{recorded or 'basic'}'; "
            f"-c cannot change it to '{args.loops or 'basic'}' (drop --loops, or -o to start over)."
        )
    else:
        args.loops = recorded

    agent_name = args.agent_spec.split(":")[0]
    # After the workspace checks, so a refused run leaves the agent's prompts alone.
    if continuing and (output_path / LAST_PROMPTS_DIR).is_dir():
        # The prompts on disk may since have been restored (the scripts, git): -c continues
        # from the optimizer's last prompts, also with --reset, which then applies only to
        # a run that has nothing to continue.
        restore_last_prompts(output_path, agent_name)
    elif args.reset:
        from agents_gym_core.prompt_optimization.reset_prompts import reset_prompts

        reset_prompts(agent_name, directory_provider=directory_provider())
        print()

    os.environ[TARGET_ENV] = args.agent_spec
    os.environ[EVAL_MODEL_ENV] = str(args.model)
    os.environ[EVAL_LOOPS_ENV] = str(args.loops)
    os.environ["COACH_OUTPUT_DIR"] = output_dir
    init_started_experiments(output_path)
    os.environ["COACH_MAX_CHARS"] = str(args.max_chars)

    # Dispatch. A run that ends early keeps the optimizer's last prompts (saved in the
    # workspace, which -c restores); only the best round so far is reported.
    save_last_prompts(output_path, agent_name)
    finished = False
    try:
        if start_round > args.rounds:
            print(f"Rounds 0..{args.rounds} of {output_dir}/ are scored.")
        elif args.engine == "claude":
            run_claude(args, output_dir, start_round)
        elif args.engine == "react":
            from langgraph.errors import GraphRecursionError

            stopped = False
            try:
                asyncio.run(
                    run_react(
                        args, output_dir, start_round=start_round, resume=args.mode == "continue"
                    )
                )
            except* GraphRecursionError:
                stopped = True
            if stopped and detect_last_completed_react_round(output_path) < args.rounds:
                sys.exit(
                    f"Stopped at --recursion-limit {args.recursion_limit}. Completed rounds are in "
                    f"{output_dir}/; -c continues the run."
                )
        finished = True
    finally:
        with suppress(OSError, UnicodeError):  # never hides how the run ended
            save_last_prompts(output_path, agent_name)
        if not finished:
            report_best_round(output_path)

    restore_best_round(output_path, agent_name)


def main(
    argv: list[str] | None = None,
    *,
    directory_provider: ProjectDirectoryProvider | None = None,
    prog: str | None = None,
) -> None:
    if directory_provider is not None and type(directory_provider) is not ProjectDirectoryProvider:
        raise TypeError("directory_provider must be an exact ProjectDirectoryProvider or None")
    with optimization_context(directory_provider):
        _main(argv, prog=prog)


if __name__ == "__main__":
    main()
