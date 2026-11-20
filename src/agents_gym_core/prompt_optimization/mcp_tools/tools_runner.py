"""MCP tools for running agents through the testbed (async with progress tracking)."""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_MAX_DATAPOINTS,
    DEFAULT_MODEL,
    EVAL_LOOPS_ENV,
    EVAL_MODEL_ENV,
    EXPERIMENT_NAME,
    directory_provider,
    experiments_dir,
    load_json_safe,
    optimization_target,
    parse_spec,
    record_started_experiment,
    repo_root,
)
from agents_gym_core.prompt_optimization.mcp_tools.usage_tracker import (
    record_run,
    update_run_results,
)
from agents_gym_core.runtime.cost_tracking import COST_FILENAME, CostTotals, load_cost_record

# ---------------------------------------------------------------------------
# In-memory registry of active/completed runs
# ---------------------------------------------------------------------------

_runs: dict[str, dict] = {}
_run_counter = 0
_naming_lock = threading.Lock()
_reserved_names: set[str] = set()


def _next_run_id() -> str:
    global _run_counter
    with _naming_lock:
        _run_counter += 1
        return f"run_{_run_counter}"


def _next_experiment_name(base_name: str) -> str:
    """Return a unique experiment name by appending _NNN based on existing folders.

    Scans experiments/ for directories matching ``{base_name}_NNN`` and the
    in-process reserved set, then returns the next available number.
    First run produces ``{base_name}_001``.  The name is reserved by creating
    its (empty) experiment folder, which fails if it already exists, so two
    calls never return the same suffix, even from two processes (concurrent
    optimizations); the testbed then writes into that folder.
    """
    with _naming_lock:
        max_num = 0
        # Ignore case: on a case-insensitive filesystem FOO_001 is the folder foo_001.
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$", re.IGNORECASE)
        if experiments_dir().is_dir():
            for d in experiments_dir().iterdir():
                if d.is_dir():
                    m = pattern.match(d.name)
                    if m:
                        max_num = max(max_num, int(m.group(1)))
        for name in _reserved_names:
            m = pattern.match(name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        experiments_dir().mkdir(parents=True, exist_ok=True)
        while True:
            max_num += 1
            result = f"{base_name}_{max_num:03d}"
            try:
                (experiments_dir() / result).mkdir()
            except FileExistsError:  # taken since the scan, e.g. by another process
                continue
            _reserved_names.add(result)
            return result


def _release_experiment_name(name: str) -> None:
    """Give back *name* (a run that did not start): drop it from the reserved set and
    remove its experiment folder if it is still empty."""
    with _naming_lock:
        _reserved_names.discard(name)
        try:
            (experiments_dir() / name).rmdir()
        except OSError:  # gone, or the testbed wrote into it
            pass


def _build_cmd(
    agent_list: list[str],
    max_datapoints: int,
    model: str,
    experiment_name: str,
    loops: list[str] | None = None,
    datapoint_list: list[str] | None = None,
) -> list[str]:
    provider = directory_provider()
    cmd = [sys.executable, "-m", "agents_gym_core.testbed_core"]
    if isinstance(provider, ProjectDirectoryProvider):
        cmd = [
            sys.executable,
            "-m",
            "agents_gym_core.sdk.cli",
            "--project",
            str(provider.get_project_root()),
            "eval",
        ]
    cmd += [
        "--agents",
        *agent_list,
        f"--max_datapoints={max_datapoints}",
        f"--models={model}",
        f"--experiment_name={experiment_name}",
        "--max_concurrency=50",
    ]
    if loops:
        cmd.extend(["--loops", *loops])
    if datapoint_list:
        cmd.extend(["--datapoints", *datapoint_list])
    return cmd


def _parse_progress(output: str) -> str:
    """Parse the latest output to determine the current phase and progress."""
    lines = output.strip().splitlines()
    if not lines:
        return "starting"

    # Walk backwards to find the latest meaningful line
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "AGGREGATED SCORES" in stripped:
            return "completed"
        if "Saved aggregated results" in stripped:
            return "completed"
        # tqdm progress bars contain "|" and "%"
        if "|" in stripped and "%" in stripped:
            # Extract percentage from tqdm: "  60%|██████    | 3/5"
            try:
                pct = stripped.split("%")[0].strip().split()[-1]
                if "Running evaluations" in output and "Running agents" in output:
                    # Already past agent execution
                    return f"evaluating ({pct}%)"
                return f"running agents ({pct}%)"
            except (IndexError, ValueError):
                pass
        if "Running evaluations" in stripped:
            return "evaluating"
        if "Running agents" in stripped:
            return "running agents"
        if "EXPERIMENT INVENTORY" in stripped:
            return "loading"

    return "in progress"


def _count_actual_datapoints(experiment_name: str, agent_name: str) -> int:
    """Count the actual number of unique datapoints evaluated for an agent in an experiment.

    Reads combined_results.json and counts unique datapoint_name entries for the agent.
    Returns 0 if the file doesn't exist or can't be parsed.
    """
    combined_file = experiments_dir() / experiment_name / "combined_results.json"
    if not combined_file.exists():
        return 0
    try:
        combined = json.loads(combined_file.read_text())
        if not isinstance(combined, list):
            return 0
        # Count unique datapoint names for this agent (base name, strip :version)
        base_name = agent_name.split(":")[0]
        datapoint_names = set()
        for entry in combined:
            entry_agent = entry.get("agent", "")
            if entry_agent == base_name or entry_agent == agent_name:
                dp_name = entry.get("datapoint_name", "")
                if dp_name:
                    datapoint_names.add(dp_name)
        return len(datapoint_names)
    except Exception:
        return 0


def _get_agent_score(experiment_name: str, agent_name: str) -> float | None:
    """Get the final score for an agent from an experiment. Returns None if unavailable."""
    scores_file = experiments_dir() / experiment_name / "final_score_per_agent.json"
    if not scores_file.exists():
        return None
    try:
        scores = json.loads(scores_file.read_text())
        base_name = agent_name.split(":")[0]
        for loop_data in scores.values():
            for agent_key, agent_data in loop_data.items():
                if agent_key.split("__")[0] == base_name:
                    return agent_data.get("all_data", {}).get("runs_mean")
        return None
    except Exception:
        return None


def _extract_round_info(experiment_name: str) -> tuple[str | None, str | None]:
    """Extract round name and output dir from experiment name.

    Experiment names follow: {prefix}_{agent}_round_NNN_{seq} (e.g. claude_opt_* or react_opt_*)
    Returns (round_name, output_dir_prefix) or (None, None) if not a round experiment.
    """
    m = re.match(r"^(.+)_(round_\d+)_\d+$", experiment_name)
    if not m:
        return None, None
    return m.group(2), m.group(1)


def _write_round_stats(
    run_info: dict, agent_name: str, datapoints: int, score: float | None
) -> None:
    """Write round_stats.json into the round subdirectory of the optimizer workspace."""
    experiment_name = run_info.get("experiment_name", "")
    model = run_info.get("model", "")

    round_name, _ = _extract_round_info(experiment_name)
    if not round_name:
        return

    env_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip()
    if not env_dir:
        return
    output_dir = Path(env_dir) if Path(env_dir).is_absolute() else repo_root() / env_dir
    round_dir = output_dir / round_name

    if not round_dir.exists():
        round_dir.mkdir(parents=True, exist_ok=True)

    # Read previous round's score for delta calculation
    previous_score = None
    round_num = int(round_name.split("_")[1])
    if round_num > 0:
        prev_round = f"round_{round_num - 1:03d}"
        prev_stats_file = output_dir / prev_round / "round_stats.json"
        if prev_stats_file.exists():
            try:
                prev = json.loads(prev_stats_file.read_text())
                previous_score = prev.get("score")
            except Exception:
                pass

    round_stats = {
        "round": round_name,
        "experiment_name": experiment_name,
        "model": model,
        "datapoints": datapoints,
        "score": round(score, 4) if score is not None else None,
    }
    if previous_score is not None and score is not None:
        round_stats["previous_score"] = previous_score
        round_stats["delta"] = round(score - previous_score, 4)

    # Every evaluation run of the round: when it ran and what its rollouts cost.
    stats_file = round_dir / "round_stats.json"
    record = _eval_run_record(run_info)
    with _round_stats_lock:  # runs finish on the MCP server's worker threads
        earlier, err = load_json_safe(stats_file) if stats_file.exists() else ({}, None)
        runs = [
            r
            for r in (earlier.get("runs", []) if not err and isinstance(earlier, dict) else [])
            if isinstance(r, dict) and r.get("experiment_name") != experiment_name
        ]
        runs.append(record)
        if run_info.get("max_datapoints") or str(run_info.get("datapoints") or "").strip():
            # A partial run is recorded but keeps the score of the round's full run, if any.
            keep = earlier if not err and isinstance(earlier, dict) else {}
            for key in (
                "experiment_name",
                "model",
                "datapoints",
                "score",
                "previous_score",
                "delta",
            ):
                round_stats.pop(key, None)
                if key in keep:
                    round_stats[key] = keep[key]
        elif run_info.get("round_prompts") is not None:
            from agents_gym_core.prompt_optimization.best_round import write_prompts

            # This run's score becomes the round's, so its prompts do too.
            write_prompts(round_dir, run_info["round_prompts"])
        round_stats["runs"] = runs
        round_stats["eval_duration_s"] = round(sum(r.get("duration_s") or 0 for r in runs), 1)
        round_stats["eval_cost_usd"] = round(
            sum((r.get("eval_cost") or {}).get("total_cost_usd") or 0 for r in runs), 6
        )
        tmp = stats_file.with_name(stats_file.name + ".tmp")
        tmp.write_text(json.dumps(round_stats, indent=2) + "\n")
        tmp.replace(stats_file)


_round_stats_lock = threading.Lock()


def _eval_run_record(run_info: dict) -> dict:
    """When an evaluation run started and finished (its scores file), and its rollouts' cost."""
    experiment_name = run_info.get("experiment_name", "")
    exp_dir = experiments_dir() / experiment_name
    scores_file = exp_dir / "final_score_per_agent.json"
    finished = scores_file.stat().st_mtime if scores_file.exists() else time.time()
    started = run_info.get("started_at", finished)
    cost = CostTotals()
    for cost_file in exp_dir.rglob(COST_FILENAME):
        if record := load_cost_record(cost_file.parent):
            cost.merge(record)
    return {
        "experiment_name": experiment_name,
        "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
        "finished_at": datetime.fromtimestamp(finished, UTC).isoformat(),
        "duration_s": round(finished - started, 1),
        "eval_cost": cost.model_dump(),
    }


def _update_actual_datapoints_for_run(run_info: dict) -> None:
    """After a run completes, count actual datapoints, get score, update stats.json and round_stats.json."""
    experiment_name = run_info.get("experiment_name", "")
    agents_str = run_info.get("agents", "")
    if not experiment_name or not agents_str:
        return
    for agent_name in agents_str.split():
        actual = _count_actual_datapoints(experiment_name, agent_name)
        score = _get_agent_score(experiment_name, agent_name)
        if actual > 0:
            update_run_results(experiment_name, agent_name, actual, score)
            _write_round_stats(run_info, agent_name, actual, score)


def _immutable_rounds_before() -> tuple[int, str | None]:
    """The first round a ``-c`` continuation may write (earlier ones are frozen), or a refusal."""
    text = os.environ.get("COACH_IMMUTABLE_ROUNDS_BEFORE", "0")
    try:
        return max(0, int(text)), None
    except ValueError:
        return 0, (
            f"Refused: COACH_IMMUTABLE_ROUNDS_BEFORE must be a non-negative integer, got '{text}'."
        )


def loop_specs(loops: str) -> set[tuple[str, str]]:
    """The (loop, version) pairs a testbed run with ``--loops loops`` evaluates, parsed as
    ``TestbedRunnerCore._parse_loops`` does, with its default ``basic`` for no loops.

    A loop without a version runs ``agents_gym_core.loops.types.DEFAULT_LOOP_VERSION``
    ("auto"), spelled out here so the MCP server does not import the loops package.
    """
    specs = set()
    for spec in loops.split() or ["basic"]:
        name, sep, version = spec.partition(":")
        specs.add((name, version if sep else "auto"))
    return specs


# ---------------------------------------------------------------------------
# Async run management
# ---------------------------------------------------------------------------


def run_agent_async(
    agents: str,
    max_datapoints: int = DEFAULT_MAX_DATAPOINTS,
    model: str = DEFAULT_MODEL,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    loops: str = "",
    datapoints: str = "",
) -> str:
    """Start an agent run asynchronously. Returns a run_id to check progress.

    Args:
        agents: Space-separated agent specs
        max_datapoints: Max datapoints per agent (0 = full eval-set, default: 0). Ignored when datapoints is provided.
        model: Model to use (default: DEFAULT_MODEL)
        experiment_name: Experiment name for output (default: DEFAULT_EXPERIMENT_NAME)
        loops: Space-separated loop specs (e.g. "deep-agents" or "basic:any planning"). Empty = default (basic).
        datapoints: Pipe-separated datapoint names to run (exact match filter), e.g. "Name one|Name two|Name three". Use pipe (|) as separator since datapoint names may contain spaces. Preferred over max_datapoints for targeted debugging. Empty = all datapoints.
    """
    agent_list = agents.split()
    if not agent_list:
        return "No agents specified."
    target = optimization_target()
    if target and any(parse_spec(spec) != target for spec in agent_list):
        return f"Refused: this optimization run evaluates only '{target[0]}:{target[1]}'."
    eval_model = os.environ.get(EVAL_MODEL_ENV, "").strip()
    if target and eval_model and model != eval_model:
        # Every round is compared on train, so all of them run on the same model.
        return f"Refused: this optimization run evaluates with model '{eval_model}'."
    eval_loops = os.environ.get(EVAL_LOOPS_ENV)  # set even when empty (the default loop)
    if target and eval_loops is not None and loop_specs(loops) != loop_specs(eval_loops):
        # ... and with the same loops.
        return f"Refused: this optimization run evaluates with loops '{eval_loops or 'basic'}'."

    loop_list = loops.split() if loops.strip() else None
    datapoint_list = (
        [d.strip() for d in datapoints.split("|") if d.strip()] if datapoints.strip() else None
    )
    if any(t.startswith("-") for t in [*agent_list, *(loop_list or []), *(datapoint_list or [])]):
        return "Refused: agent, loop and datapoint names cannot start with '-'."
    if not EXPERIMENT_NAME.fullmatch(experiment_name):
        return f"Refused: experiment_name may only use letters, digits and _.:+- ({experiment_name!r})."
    experiment_name = _next_experiment_name(experiment_name)
    round_name, _ = _extract_round_info(experiment_name)
    if round_name:
        # A run rewrites its round's round_stats.json and prompt snapshot.
        immutable_before, refused = _immutable_rounds_before()
        round_num = int(round_name.split("_")[1])
        if not refused and round_num < immutable_before:
            refused = (
                f"Refused: round_{round_num:03d} is immutable in this continuation "
                f"(new runs must target round_{immutable_before:03d} or later)."
            )
        if refused:
            _release_experiment_name(experiment_name)
            return refused
    env_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip()
    full_run = not max_datapoints and not datapoints.strip()
    round_prompts = None
    if target and round_name and env_dir and full_run:  # a partial run keeps the round's snapshot
        from agents_gym_core.prompt_optimization.best_round import read_prompts

        # Saved as the round's prompts only when this run's score is recorded.
        round_prompts = read_prompts(target[0])
    cmd = _build_cmd(agent_list, max_datapoints, model, experiment_name, loop_list, datapoint_list)
    run_id = _next_run_id()

    # Create temp files for stdout/stderr
    stdout_file = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"coach_{run_id}_out_", suffix=".log", delete=False
    )
    stderr_file = tempfile.NamedTemporaryFile(
        mode="w", prefix=f"coach_{run_id}_err_", suffix=".log", delete=False
    )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            cwd=str(repo_root()),
        )
    except Exception as e:
        stdout_file.close()
        stderr_file.close()
        _release_experiment_name(experiment_name)
        return f"Error starting agent run: {e}"

    _runs[run_id] = {
        "process": process,
        "stdout_path": stdout_file.name,
        "stderr_path": stderr_file.name,
        "agents": agents,
        "experiment_name": experiment_name,
        "max_datapoints": max_datapoints,
        "datapoints": datapoints,
        "model": model,
        "started_at": time.time(),
        "round_prompts": round_prompts,
    }

    record_started_experiment(experiment_name)
    record_run(agent_list, max_datapoints, model, experiment_name, datapoints=datapoint_list)

    if datapoints.strip():
        scope_info = f"Datapoints: {datapoints}\n"
    else:
        scope_info = f"Max datapoints: {max_datapoints}\n" if max_datapoints > 0 else ""
    return (
        f"Started run '{run_id}'\n"
        f"Agents: {agents}\n"
        f"Model: {model}\n"
        f"{scope_info}"
        f"Experiment: {experiment_name}\n\n"
        f'Use get_run_status(run_id="{run_id}") to check progress.'
    )


def get_run_status(run_id: str) -> str:
    """Check the status and progress of an async agent run.

    Args:
        run_id: The run ID returned by run_agent_async
    """
    if run_id not in _runs:
        active = [rid for rid, r in _runs.items()]
        return f"Run '{run_id}' not found. Active runs: {active or 'none'}"

    run = _runs[run_id]
    if "final_status" in run:
        return run["final_status"]
    process = run["process"]
    elapsed = time.time() - run["started_at"]
    elapsed_str = f"{elapsed:.0f}s"

    # Read current output
    stdout_content = ""
    try:
        stdout_content = Path(run["stdout_path"]).read_text()
    except Exception:
        pass

    stderr_content = ""
    try:
        stderr_content = Path(run["stderr_path"]).read_text()
    except Exception:
        pass

    returncode = process.poll()
    phase = _parse_progress(stdout_content + "\n" + stderr_content)

    if returncode is None:
        # Still running
        return (
            f"Run '{run_id}' — RUNNING ({elapsed_str} elapsed)\n"
            f"Agents: {run['agents']}\n"
            f"Phase: {phase}\n"
            f"Experiment: {run['experiment_name']}\n\n"
            f'Use get_run_status(run_id="{run_id}") to check again.\n'
            f'When done, use get_full_results(experiment_name="{run["experiment_name"]}") to see results.'
        )

    # The testbed exits 1 after listing failed datapoint runs or raised evaluators, but it still
    # writes its scores (failed runs left out of the mean, raised evaluators counted 0): such a
    # run is scored and recorded like a clean one.
    scored = returncode == 0 or (
        returncode == 1
        and (experiments_dir() / run["experiment_name"] / "final_score_per_agent.json").is_file()
    )
    status = "COMPLETED" if scored else f"FAILED (exit code {returncode})"

    # Count actual datapoints, score, and write round_stats.json
    if scored:
        _update_actual_datapoints_for_run(run)

    # Clean up temp files
    for key in ("stdout_path", "stderr_path"):
        try:
            Path(run[key]).unlink(missing_ok=True)
        except Exception:
            pass

    scores_section = _extract_scores_section(stdout_content)
    if not scored:
        stderr_lines = [
            line
            for line in stderr_content.splitlines()
            if line.strip() and "it/s]" not in line and "s/it]" not in line
        ]
        if stderr_lines:
            scores_section += "\n\nSTDERR (last 40 lines):\n" + "\n".join(stderr_lines[-40:])

    run["final_status"] = (
        f"Run '{run_id}' — {status} ({elapsed_str})\n"
        f"Agents: {run['agents']}\n\n"
        f"{scores_section}\n\n"
        f'Use get_full_results(experiment_name="{run["experiment_name"]}") for detailed results.'
    )
    return run["final_status"]


def wait_for_run(run_id: str, poll_interval: int = 30) -> str:
    """Block until an async agent run completes. Returns the final status.

    Args:
        run_id: The run ID returned by run_agent_async
        poll_interval: Seconds between status checks (default: 30)
    """
    if run_id not in _runs:
        active = [rid for rid in _runs]
        return f"Run '{run_id}' not found. Active runs: {active or 'none'}"

    poll_interval = max(5, min(poll_interval, 120))

    run = _runs[run_id]
    process = run["process"]

    while process.poll() is None:
        time.sleep(poll_interval)

    return get_run_status(run_id)


# Seconds a testbed run still in flight gets to exit after SIGTERM when the server shuts down.
_TERMINATE_GRACE_S = 5.0


def terminate_active_runs() -> None:
    """Stop the testbed runs still in flight when the MCP server shuts down, so none outlives
    the optimization session: SIGTERM, then SIGKILL after a short grace. Each is recorded as
    terminated."""
    active = {
        run_id: run
        for run_id, run in list(_runs.items())
        if "final_status" not in run and run["process"].poll() is None
    }
    for run_id, run in active.items():
        run["final_status"] = (
            f"Run '{run_id}' — TERMINATED (the MCP server shut down)\n"
            f"Agents: {run['agents']}\n"
            f"Experiment: {run['experiment_name']}"
        )
        try:
            run["process"].terminate()
        except OSError:
            pass
    deadline = time.monotonic() + _TERMINATE_GRACE_S
    for run_id, run in active.items():
        process = run["process"]
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            try:
                process.kill()
                process.wait()
            except OSError:
                pass
        for key in ("stdout_path", "stderr_path"):
            try:
                Path(run[key]).unlink(missing_ok=True)
            except OSError:
                pass
        print(
            f"coach-light: terminated {run_id} ({run['experiment_name']}) at shutdown",
            file=sys.stderr,
        )


# Other files gym optimize writes in the workspace: the best round (best_round.py), the run's
# metadata, the claude engine's MCP config, the usage stats (usage_tracker.py), the react
# engine's thread state and checkpoint database (with SQLite's side files), and the temporary
# file round_stats.json is written through.
_RESERVED_WORKSPACE_NAMES = {
    "best_round.json",
    "run_metadata.json",
    "mcp_config.json",
    "stats.json",
    "react_thread.json",
    "react_checkpoints.sqlite3",
    "react_checkpoints.sqlite3-wal",
    "react_checkpoints.sqlite3-shm",
    "react_checkpoints.sqlite3-journal",
    "round_stats.json.tmp",
}


def save_optimizer_log(output_dir: str, filename: str, content: str) -> str:
    """Save a log file to the optimizer output directory.

    Uses COACH_OUTPUT_DIR env var if set (optimizer launcher), otherwise uses the output_dir parameter
    (interactive skill usage).

    Args:
        output_dir: The output directory name (e.g. "tmp_claude_optimizer_agent_model")
        filename: The filename to write, supports subdirs (e.g. "round_000/scores.md")
        content: The file content
    """
    output_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip() or output_dir

    if not re.match(r"^tmp_[\w:.-]+$", output_dir):
        return f"Refused: output_dir '{output_dir}' does not match allowed pattern 'tmp_*'."

    if "\\" in filename or ".." in filename:
        return f"Refused: filename '{filename}' must not contain '..' or backslashes."
    if any(part.startswith(".") for part in Path(filename).parts):
        return f"Refused: filename '{filename}' must not name a hidden file."
    # As any part of the path: a folder of that name would also break the file's writer.
    parts = [part.casefold().rstrip(" .") for part in Path(filename).parts]
    if "round_stats.json" in parts:
        return "Refused: round_stats.json is written by run_agent_async."
    if "evaluated_prompts" in parts:
        return "Refused: evaluated_prompts/ is written by run_agent_async."
    if reserved := next((part for part in parts if part in _RESERVED_WORKSPACE_NAMES), None):
        return f"Refused: {reserved} is written by gym optimize."

    immutable_before, refused = _immutable_rounds_before()
    if refused:
        return refused
    target_dir = repo_root() / output_dir
    target_file = target_dir / filename

    try:
        relative = target_file.resolve().relative_to(target_dir.resolve())
    except ValueError:
        return f"Refused: filename '{filename}' escapes the output directory."

    # Check on the normalized path, so spellings such as "./round_000/scores.md" or an absolute
    # path cannot alias an immutable round or the session records.
    if relative.parts and relative.parts[0].casefold().rstrip(" .") == "sessions":
        return "Refused: sessions/ is written by gym optimize (it records run time and cost)."
    # Casefolded: on a case-insensitive filesystem ROUND_000/ is the folder round_000/.
    first = relative.parts[0].casefold().rstrip(" .") if relative.parts else ""
    round_match = re.fullmatch(r"round_(\d+)", first)
    if round_match and int(round_match.group(1)) < immutable_before:
        return (
            f"Refused: round_{int(round_match.group(1)):03d} is immutable in this "
            f"continuation (new writes must target round_{immutable_before:03d} or later)."
        )

    if target_file.exists():
        return f"Refused: {relative} is already saved; optimizer logs are write-once."
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content)
    return f"Saved {target_file.relative_to(repo_root())} ({len(content)} chars)"


def read_optimizer_log(output_dir: str, filename: str) -> str:
    """Read a file from the optimizer output directory.

    Uses COACH_OUTPUT_DIR env var if set (optimizer launcher), otherwise uses the output_dir parameter
    (interactive skill usage).

    Args:
        output_dir: The output directory name (e.g. "tmp_claude_optimizer_agent_model")
        filename: The filename to read (e.g. "round_000/system_prompt.md")
    """
    output_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip() or output_dir

    if not re.match(r"^tmp_[\w:.-]+$", output_dir):
        return f"Refused: output_dir '{output_dir}' does not match allowed pattern 'tmp_*'."

    if "\\" in filename or ".." in filename:
        return f"Refused: filename '{filename}' must not contain '..' or backslashes."

    target_dir = repo_root() / output_dir
    target_file = target_dir / filename

    if not target_file.resolve().is_relative_to(target_dir.resolve()):
        return f"Refused: filename '{filename}' escapes the output directory."

    if not target_file.exists():
        available = [
            str(f.relative_to(target_dir)) for f in sorted(target_dir.rglob("*")) if f.is_file()
        ]
        return f"File not found: {filename}. Available: {available}"

    content = target_file.read_text()
    return f"=== {filename} ({len(content)} chars) ===\n\n{content}"


def _extract_scores_section(output: str) -> str:
    """Extract the AGGREGATED SCORES SUMMARY section from testbed output."""
    lines = output.splitlines()
    in_scores = False
    scores_lines = []
    for line in lines:
        if "AGGREGATED SCORES SUMMARY" in line:
            in_scores = True
        if in_scores:
            scores_lines.append(line)
            if line.startswith("===") and len(scores_lines) > 2:
                break
    return "\n".join(scores_lines) if scores_lines else output[-1000:]
