"""Optimization protocol for the ReAct prompt optimizer.

Loads a skill prompt from ``skills/<name>.md`` without the sentence on resuming from a given
round (and without a YAML preamble, if a file has one). The skills are the
paper's: ``full`` (sigma_Full, the default), ``trimmed`` (sigma_Trimmed, the skill prompt
ablation) and ``tool-free`` (sigma_Ablate, the simplified skill of the score-only and macro
tools ablations).
"""

import os
import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
DEFAULT_SKILL = "full"

SYSTEM_PROMPT = (
    "You are a prompt optimization agent. Execute the optimization protocol autonomously. "
    "Do NOT ask the user questions — just work through all rounds without stopping."
)


def skill_names() -> list[str]:
    return sorted(f.stem for f in SKILLS_DIR.glob("*.md"))


def _load_skill(name: str = DEFAULT_SKILL) -> str:
    """Load a skill file without its continuation sentence (or any YAML frontmatter)."""
    path = SKILLS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Skill '{name}' not found; available: {', '.join(skill_names())}")
    content = path.read_text()
    # Strip YAML frontmatter (--- ... ---)
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    # Strip continuation sentence
    content = content.replace(
        " If a starting round is provided (e.g. continuing from a previous session),"
        " begin from that round instead of 0.",
        "",
    )
    return content.strip()


def build_optimization_prompt(
    agent_name: str,
    model: str,
    num_rounds: int,
    output_dir: str,
    skill_version: str = DEFAULT_SKILL,
    loops: str = "",
    optimizer_model: str = "",
    start_round: int = 0,
) -> str:
    """Build the optimization protocol prompt from the versioned skill file.

    Substitutes $ARGUMENTS[0..4] and the experiment base name.
    """
    content = _load_skill(skill_version)
    if start_round > 0:
        content = (
            "# CONTINUATION OVERRIDE\n\n"
            f"Rounds 0 through {start_round - 1} are already complete in "
            f"`{output_dir}`. Do NOT rerun or re-analyze Round 0 as a new "
            f"baseline. Begin at Round {start_round} by reading the prior round's "
            "saved scores and prompts. Before starting the next evaluation, update "
            "at least one existing configured prompt such as a signature prompt; "
            "do not attempt to create a system prompt when agent.yaml has none. "
            f"Continue through Round {num_rounds}, inclusive.\n\n" + content
        )

    # Substitute $ARGUMENTS[0..4]
    content = content.replace("$ARGUMENTS[0]", agent_name)
    content = content.replace("$ARGUMENTS[1]", model)
    content = content.replace("$ARGUMENTS[2]", str(num_rounds))
    content = content.replace("$ARGUMENTS[3]", output_dir)
    content = content.replace("$ARGUMENTS[4]", loops)

    exp_prefix = build_experiment_prefix(
        skill_version=skill_version,
        optimizer_model=optimizer_model,
    )
    content = content.replace("claude_opt", exp_prefix)

    return content


def build_experiment_prefix(
    skill_version: str,
    optimizer_model: str,
) -> str:
    """Return the experiment prefix embedded in the optimizer protocol."""
    temp = os.environ.get("UIPATH_AGENTS_GYM_TEMPERATURE", "1.0")
    opt_short = re.sub(r"[:/]+", "_", optimizer_model) if optimizer_model else "unknown"
    return f"react_opt_{skill_version}_t{temp}_{opt_short}"
