"""Time and cost of an optimization run, summed over the sessions of its workspace.

Each session (a run and each ``-c`` continuation) writes ``sessions/<id>.json`` with its
``started_at``, ``duration_s`` and ``optimizer_cost``; the evaluation runs it started are its
rollouts. ``run_totals`` adds them up for ``run_metadata.json``.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    experiments_dir,
    started_experiments,
)
from agents_gym_core.runtime.cost_tracking import COST_FILENAME, CostTotals, load_cost_record

SESSIONS_DIRNAME = "sessions"


def optimizer_cost(output_path: Path) -> dict[str, Any]:
    """The optimizer model's usage summed over the workspace's sessions (``-c`` adds one)."""
    total = CostTotals()
    for session in sorted((output_path / SESSIONS_DIRNAME).glob("*.json")):
        try:
            data = json.loads(session.read_text())
        except (OSError, ValueError):
            continue  # a session cut off while writing its file
        if isinstance(data, dict) and (cost := data.get("optimizer_cost")):
            total.merge(CostTotals.model_validate(cost))
    return total.model_dump()


def eval_cost() -> dict[str, Any]:
    """The evaluated agents' usage in the experiments this run started (its rollouts)."""
    total = CostTotals()
    for name in sorted(started_experiments()):
        for cost_file in (experiments_dir() / name).rglob(COST_FILENAME):
            # Skips a cost.json that is empty or cut off (e.g. a run stopped while writing it).
            if record := load_cost_record(cost_file.parent):
                total.merge(record)
    return total.model_dump()


def sessions(output_path: Path) -> list[dict[str, Any]]:
    out = []
    for session in sorted((output_path / SESSIONS_DIRNAME).glob("*.json")):
        try:
            data = json.loads(session.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def timing(output_path: Path) -> dict[str, Any]:
    """Time spent over all sessions, and when the best round's evaluation finished."""
    recorded = sessions(output_path)
    timing: dict[str, Any] = {
        "total_duration_s": round(sum(s.get("duration_s") or 0 for s in recorded), 1)
    }
    rounds = []
    for stats_file in sorted(output_path.glob("round_*/round_stats.json")):
        try:
            stats = json.loads(stats_file.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(stats, dict) and isinstance(stats.get("score"), (int, float)):
            rounds.append(stats)
    starts = [s["started_at"] for s in recorded if s.get("started_at")]
    if rounds and starts:
        best = max(rounds, key=lambda r: r["score"])  # the earliest round among equal scores
        # The run that gave the round its score, not a later partial run of the round.
        runs = best.get("runs") or [{}]
        scored = [r for r in runs if r.get("experiment_name") == best.get("experiment_name")]
        finished = (scored or runs)[-1].get("finished_at")
        timing |= {"best_round": best.get("round"), "best_score": best["score"]}
        if finished:
            elapsed = datetime.fromisoformat(finished) - datetime.fromisoformat(min(starts))
            timing["time_to_best_round_s"] = round(elapsed.total_seconds(), 1)
    return timing


def print_run_summary(totals: dict[str, Any]) -> None:
    opt, ev = totals["optimizer_cost"], totals["eval_cost"]
    best, to_best = totals.get("best_round"), totals.get("time_to_best_round_s")
    print(
        f"  Time:      {totals['total_duration_s']:.0f}s"
        + (f" (best: {best}, after {to_best:.0f}s)" if best and to_best is not None else "")
        + f"\n  Optimizer: {opt['input_tokens']:,} in / {opt['output_tokens']:,} out tokens,"
        f" ${opt['total_cost_usd']:.2f}"
        + f"\n  Rollouts:  {ev['input_tokens']:,} in / {ev['output_tokens']:,} out tokens,"
        f" ${ev['total_cost_usd']:.2f}" + f"\n  Total:     ${totals['total_cost_usd']:.2f}"
    )


def run_totals(output_path: Path) -> dict[str, Any]:
    """Optimizer and rollout cost, their sum and the run's timing; on failure, the error."""
    try:
        totals: dict[str, Any] = {
            "optimizer_cost": optimizer_cost(output_path),
            "eval_cost": eval_cost(),
            **timing(output_path),
        }
        totals["total_cost_usd"] = round(
            totals["optimizer_cost"]["total_cost_usd"] + totals["eval_cost"]["total_cost_usd"], 6
        )
    except Exception as exc:  # accounting must never hide how the run ended
        return {"accounting_error": f"{type(exc).__name__}: {exc}"}
    return totals
