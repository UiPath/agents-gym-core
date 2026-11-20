"""Persistent usage tracking for the optimizer's MCP server.

Tracks per-agent:
- System/user prompt version history (v1, v2, v3, ...)
- Run ledger: each run records which prompt versions + model were active

Storage: ``stats.json`` in the active optimization workspace, alongside round directories
and session logs. ``COACH_OUTPUT_DIR`` selects the workspace used by both optimizer engines;
standalone helpers resolve an agent/model directory through ``_output_dir``.

The active model per agent is remembered in memory so that
record_prompt_update (which only knows the agent name) can write to the
correct model-specific output dir.

Per-agent file schema:
{
  "system_prompt": {
    "current_version": "v3",
    "history": [
      {"version": "v1", "updated_at": "<iso>"},
      ...
    ]
  },
  "user_prompt": { ... },
  "rounds": [
    {
      "experiment_name": "react_opt_agent_round_000_001",
      "system_prompt": "v2",
      "user_prompt": "v1",
      "model": "gpt-5-2025-08-07",
      "datapoints": 26,
      "score": 0.924,
      "targeted": ["dp1", "dp2"],
      "started_at": "<iso>"
    }
  ]
}
"""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents_gym_core.prompt_optimization.mcp_tools.helpers import repo_root

_lock = threading.Lock()

# In-memory map: agent_name → model, set by record_run so that
# record_prompt_update can resolve the correct output dir.
_active_model: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _output_dir(agent_name: str, model: str) -> Path:
    """Return the optimizer output directory for an agent+model pair.

    Priority:
    1. COACH_OUTPUT_DIR (set by the optimizers), so the labelled workspace is always used.
    2. The first existing tmp_* dir under the repo root (in directory-listing order) whose
       name ends with _{agent_name}_{model} (interactive use). Optimizer workspaces
       (``compute_output_dir``) end with ``_temp_{T}``, so it does not match them.
    3. Default to tmp_optimizer_{agent_name}_{model}.
    """
    env_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = repo_root() / p
        return p

    suffix = f"_{agent_name}_{model}"
    matching = [
        d
        for d in repo_root().iterdir()
        if d.is_dir() and d.name.startswith("tmp_") and d.name.endswith(suffix)
    ]
    if matching:
        return matching[0]
    return repo_root() / f"tmp_optimizer_{agent_name}_{model}"


def _tracker_file_for(agent_name: str, model: str) -> Path:
    """Return the path to the usage stats file for an agent+model pair."""
    if not agent_name or "/" in agent_name or ".." in agent_name:
        raise ValueError(f"Invalid agent name: {agent_name!r}")
    d = _output_dir(agent_name, model)
    d.mkdir(parents=True, exist_ok=True)
    return d / "stats.json"


def _resolve_tracker_file(agent_name: str) -> Path:
    """Resolve the tracker file using the in-memory active model.

    ``record_run()`` updates the active model used by subsequent prompt-history entries.

    When COACH_OUTPUT_DIR is set the directory is already pinned, so the
    _active_model check is bypassed entirely.
    """
    if not agent_name or "/" in agent_name or ".." in agent_name:
        raise ValueError(f"Invalid agent name: {agent_name!r}")

    # Env var pins the directory — no need to know the model.
    env_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip()
    if env_dir:
        return _tracker_file_for(agent_name, "")

    model = _active_model.get(agent_name)
    if not model:
        raise ValueError(
            f"No active model for '{agent_name}'. "
            f"Run the agent first (run_agent_async) so the model is known."
        )
    return _tracker_file_for(agent_name, model)


def _empty_agent() -> dict[str, Any]:
    """Return a blank per-agent stats structure."""
    return {
        "system_prompt": {"current_version": None, "history": []},
        "user_prompt": {"current_version": None, "history": []},
        "rounds": [],
    }


def _load(path: Path) -> dict[str, Any]:
    """Load a usage stats file, returning an empty structure if missing or corrupt."""
    if not path.exists():
        return _empty_agent()
    try:
        data = json.loads(path.read_text())
        data.setdefault("system_prompt", {"current_version": None, "history": []})
        data.setdefault("user_prompt", {"current_version": None, "history": []})
        data.setdefault("rounds", [])
        for key in ("system_prompt", "user_prompt"):
            if isinstance(data[key], dict):
                data[key].setdefault("current_version", None)
                data[key].setdefault("history", [])
        if not isinstance(data["rounds"], list):
            data["rounds"] = []
        return data
    except Exception:
        return _empty_agent()


def _save(path: Path, data: dict[str, Any]) -> None:
    """Write usage stats to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _next_version(history: list[dict]) -> str:
    return f"v{len(history) + 1}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_prompt_update(agent_name: str, prompt_type: str, round_name: str = "") -> None:
    """Record that a prompt was updated — uses round name as version label."""
    now = datetime.now(UTC).isoformat()
    key = "system_prompt" if prompt_type == "system" else "user_prompt"
    try:
        with _lock:
            path = _resolve_tracker_file(agent_name)
            data = _load(path)
            prompt_entry = data[key]
            version = round_name if round_name else _next_version(prompt_entry["history"])
            prompt_entry["history"].append({"version": version, "updated_at": now})
            prompt_entry["current_version"] = version
            _save(path, data)
    except (ValueError, Exception):
        pass


def record_run(
    agent_names: list[str],
    max_datapoints: int,
    model: str,
    experiment_name: str = "mcp_run",
    datapoints: list[str] | None = None,
) -> None:
    """Record that a run was started — captures active prompt versions at this moment."""
    now = datetime.now(UTC).isoformat()

    with _lock:
        for name in agent_names:
            base_name = name.split(":")[0]
            try:
                # Remember the model for this agent so prompt updates go to the right dir
                _active_model[base_name] = model
                path = _tracker_file_for(base_name, model)
                data = _load(path)
            except ValueError:
                continue

            current_sys = data["system_prompt"]["current_version"]
            current_usr = data["user_prompt"]["current_version"]

            run_entry: dict[str, Any] = {
                "experiment_name": experiment_name,
                "system_prompt": current_sys,
                "user_prompt": current_usr,
                "model": model,
                "datapoints": None,
                "score": None,
                "started_at": now,
            }
            if datapoints:
                run_entry["targeted"] = datapoints

            # Upsert: if the last round has the same prompt versions, replace it
            # (multiple runs within the same round = keep the latest)
            rounds = data["rounds"]
            if (
                rounds
                and rounds[-1].get("system_prompt") == current_sys
                and rounds[-1].get("user_prompt") == current_usr
            ):
                rounds[-1] = run_entry
            else:
                rounds.append(run_entry)
            _save(path, data)


def update_run_results(
    experiment_name: str,
    agent_name: str,
    datapoints: int,
    score: float | None = None,
) -> None:
    """Update the results for a completed run (datapoints count and score)."""
    try:
        base_name = agent_name.split(":")[0]
        with _lock:
            path = _resolve_tracker_file(base_name)
            data = _load(path)
            for run in reversed(data["rounds"]):
                if run["experiment_name"] == experiment_name:
                    run["datapoints"] = datapoints
                    if score is not None:
                        run["score"] = round(score, 4)
                    break
            _save(path, data)
    except (ValueError, Exception):
        pass


def get_stats(agent_name: str = "") -> str:
    """Return a formatted usage report, optionally filtered to a single agent."""

    def _format(label: str, entry: dict) -> str:
        lines = [f"=== {label} ==="]

        for prompt_key, heading in [
            ("system_prompt", "SYSTEM PROMPT"),
            ("user_prompt", "USER PROMPT"),
        ]:
            pe = entry.get(prompt_key, {"current_version": None, "history": []})
            history = pe.get("history", [])
            current = pe.get("current_version")
            lines.append(f"\n{heading} VERSIONS")
            if not history:
                lines.append("  No updates recorded yet.")
            else:
                lines.append(
                    f"  Current: {current}  ({len(history)} update{'s' if len(history) != 1 else ''})"
                )
                for h in history:
                    marker = "  ← current" if h["version"] == current else ""
                    ts = h["updated_at"][:16].replace("T", " ")
                    lines.append(f"  {h['version']}  →  {ts}{marker}")

        rounds = entry.get("rounds", [])
        total_actual_dp = sum(
            r.get("datapoints") or 0 for r in rounds if isinstance(r.get("datapoints"), int)
        )
        lines.append(f"\nROUNDS ({len(rounds)} total, {total_actual_dp} datapoints evaluated)")
        if not rounds:
            lines.append("  No rounds recorded yet.")
        else:
            for r in rounds:
                ts = r.get("started_at", "")[:16].replace("T", " ")
                sys_v = r.get("system_prompt") or "—"
                usr_v = r.get("user_prompt") or "—"
                dp = r.get("datapoints")
                targeted = r.get("targeted")
                if isinstance(dp, int):
                    scope = " (targeted)" if targeted else ""
                    dp_str = f"datapoints={dp}{scope}"
                else:
                    dp_str = "datapoints=pending"
                score = r.get("score")
                score_str = f"score={score * 100:.1f}%" if score is not None else ""
                lines.append(
                    f"  {r.get('experiment_name', '?')}  "
                    f"system_prompt={sys_v}  "
                    f"user_prompt={usr_v}  "
                    f"model={r.get('model', '?')}  "
                    f"{dp_str}  "
                    f"{score_str}  "
                    f"{ts}"
                )
        return "\n".join(lines)

    try:
        with _lock:
            if agent_name:
                base_name = agent_name.split(":")[0]
                try:
                    path = _resolve_tracker_file(base_name)
                except ValueError:
                    # Cold start: no run yet in this session, _active_model unpopulated — nothing to show.
                    return f"No usage recorded yet for '{base_name}'."
                if not path.exists():
                    return f"No usage recorded yet for '{base_name}'."
                data = _load(path)
                return _format(base_name, data)

            return "Please provide an agent_name to get usage stats."
    except ValueError as e:
        return f"Error: invalid agent name — {e}"


def reset_stats(agent_name: str = "", prompt_type: str = "", scope: str = "") -> str:
    """Reset usage stats with fine-grained control."""
    try:
        with _lock:
            if not agent_name:
                deleted = []
                # An optimization run (COACH_OUTPUT_DIR set) resets only its own workspace.
                if os.environ.get("COACH_OUTPUT_DIR", "").strip():
                    workspaces = [_output_dir("", "")]
                else:
                    workspaces = sorted(repo_root().iterdir())
                for d in workspaces:
                    if d.is_dir() and d.name.startswith("tmp_"):
                        p = d / "stats.json"
                        if p.exists():
                            p.unlink()
                            deleted.append(f"{d.name}/stats.json")
                if deleted:
                    return f"Reset all usage stats. Deleted: {', '.join(deleted)}"
                return "No usage stats files found to reset."

            base_name = agent_name.split(":")[0]
            try:
                path = _resolve_tracker_file(base_name)
            except ValueError:
                # Cold start: no run yet in this session, _active_model unpopulated — nothing to reset.
                return f"No stats found for '{base_name}'."

            if not path.exists():
                return f"No stats found for '{base_name}'."

            data = _load(path)

            if prompt_type:
                key = "system_prompt" if prompt_type == "system" else "user_prompt"
                data[key] = {"current_version": None, "history": []}
                _save(path, data)
                return f"Reset {prompt_type} prompt version history for '{base_name}'."

            if scope in ("rounds", "runs"):
                data["rounds"] = []
                _save(path, data)
                return f"Reset rounds ledger for '{base_name}'."

            path.unlink()
            return f"Reset all stats for '{base_name}' (deleted {path.relative_to(repo_root())})."
    except ValueError as e:
        return f"Error: invalid agent name — {e}"
