"""Best-on-train selection for gym optimize, independent of what the optimizer model does.

When a round's full evaluation run is scored, the prompts it started with are saved with that
score; when the run ends, the agent's prompts are set to the saved prompts of the highest-scoring
round (the earliest among equal scores). The prompts the optimizer last left are saved too, so
``-c`` continues from them whatever is on disk by then.
"""

import json
from pathlib import Path

from agents_gym_core.prompt_optimization.mcp_tools.helpers import load_json_safe, require_agent_dir
from agents_gym_core.prompt_optimization.mcp_tools.tools_agents import (
    _atomic_write,
    _resolve_prompt,
    list_prompt_types,
)
from agents_gym_core.prompt_optimization.run_accounting import SESSIONS_DIRNAME

SNAPSHOT_DIR = "evaluated_prompts"
# The agent's prompts as the optimizer last left them, which -c restores (under sessions/, which
# the optimizer cannot write).
LAST_PROMPTS_DIR = Path(SESSIONS_DIRNAME) / "last_prompts"


def _prompt_files(agent_name: str) -> list[tuple[Path, Path, str]]:
    """(file, path relative to the agent dir, content) of each prompt agent.yaml configures."""
    agent_dir, err = require_agent_dir(agent_name, "all")
    if err:
        return []
    files = []
    for prompt_type in list_prompt_types(agent_name):
        path, content, err = _resolve_prompt(agent_name, prompt_type, "all")
        if err or path is None or content is None:
            continue
        file = Path(path).resolve()
        if file.is_relative_to(agent_dir.resolve()):  # update_prompt cannot edit any other
            files.append((file, file.relative_to(agent_dir.resolve()), content))
    return files


def read_prompts(agent_name: str) -> list[tuple[Path, str]]:
    """(path relative to the agent dir, content) of each prompt agent.yaml configures."""
    return [(rel, content) for _, rel, content in _prompt_files(agent_name)]


def snapshot_prompts(round_dir: Path, agent_name: str) -> None:
    """Copy the agent's current prompt files into round_dir as the round's prompts."""
    write_prompts(round_dir, read_prompts(agent_name))


def write_prompts(round_dir: Path, prompts: list[tuple[Path, str]]) -> None:
    """Write ``prompts`` (from :func:`read_prompts`) into round_dir as the round's prompts."""
    for rel, content in prompts:
        dest = round_dir / SNAPSHOT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def best_round(output_path: Path) -> tuple[Path, float] | None:
    """The highest-scoring round directory and its train score (the earliest among equal
    scores); None before any round is scored."""
    scores = {}
    for stats_file in sorted(output_path.glob("round_*/round_stats.json")):
        stats, err = load_json_safe(stats_file)
        score = stats.get("score") if not err and isinstance(stats, dict) else None
        if isinstance(score, int | float):
            scores[stats_file.parent] = score
    if not scores:
        return None
    best = max(scores, key=lambda d: scores[d])
    return best, scores[best]


def save_last_prompts(output_path: Path, agent_name: str) -> None:
    """Save the agent's current prompt files in the workspace as the optimizer's last prompts."""
    for rel, content in read_prompts(agent_name):
        dest = output_path / LAST_PROMPTS_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(dest, content)


def restore_last_prompts(output_path: Path, agent_name: str) -> None:
    """Set the agent's prompts to the optimizer's last saved prompts (for ``-c``)."""
    saved = output_path / LAST_PROMPTS_DIR
    if saved.is_dir():
        restored = _restore_prompts(saved, agent_name)
        print(f"Prompts set to the optimizer's last prompts; restored {len(restored)} file(s).")


def report_best_round(output_path: Path) -> None:
    """Print the best round so far of a run that stopped early. The optimizer's last prompts
    are saved in the workspace, and ``-c`` restores them before it continues."""
    last = (
        f"The optimizer's last prompts are saved in {output_path / LAST_PROMPTS_DIR}; -c restores "
        "them and continues from there."
    )
    found = best_round(output_path)
    if found is None:
        print(f"No round was scored. {last}")
        return
    best, score = found
    print(
        f"Best round so far: {best.name} (train {score}); its prompts are in "
        f"{best / SNAPSHOT_DIR}. {last}"
    )


def _restore_prompts(snapshot: Path, agent_name: str) -> list[str]:
    """Write the snapshot's copy of each configured prompt file that differs; anything else in
    the snapshot is ignored."""
    restored = []
    for file, rel, content in _prompt_files(agent_name):
        saved = snapshot / rel
        if saved.is_file() and not saved.is_symlink() and saved.read_text() != content:
            _atomic_write(file, saved.read_text())
            restored.append(str(rel))
    return restored


def restore_best_round(output_path: Path, agent_name: str) -> dict | None:
    """Set the agent's prompts to the best round's snapshot and record it in best_round.json."""
    found = best_round(output_path)
    if found is None:
        return None
    best, score = found
    snapshot = best / SNAPSHOT_DIR
    if not snapshot.is_dir():
        print(
            f"Best round {best.name} has no prompt snapshot; prompts left as the optimizer set them."
        )
        return None
    restored = _restore_prompts(snapshot, agent_name)
    result = {"round": best.name, "score": score, "restored": restored}
    (output_path / "best_round.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Prompts set to best round {best.name} (train {score}); restored {len(restored)} file(s)."
    )
    return result
