"""Reset an agent's prompts to their original (baseline) versions.

Copies all prompt files from prompts/baseline/ back to prompts/, restoring
the pre-optimization state. Useful before running a fresh optimization pass.

Usage:
    python -m agents_gym_core.prompt_optimization.reset_prompts <agent_name>

Examples:
    python -m agents_gym_core.prompt_optimization.reset_prompts claim_history
"""

import argparse
import re
import shutil
from pathlib import Path

from agents_gym_core.utils.directory_provider import CoreDirectoryProvider, DirectoryProvider


def find_agent_dir(
    agent_name: str, *, directory_provider: DirectoryProvider | None = None
) -> Path | None:
    """Find the agent directory in gym or core agents.

    Requires agent.yaml to exist — matches the resolution logic used by
    the MCP tools (resolve_agent_dir) so that reset writes to the same
    directory that get_prompt / update_prompt read from.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", agent_name):
        return None
    provider = directory_provider or CoreDirectoryProvider()
    for root in provider.agent_roots("all"):
        path = root / agent_name
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        if path.is_dir() and (path / "agent.yaml").is_file():
            return path
    return None


def reset_prompts(
    agent_name: str,
    dry_run: bool = False,
    *,
    directory_provider: DirectoryProvider | None = None,
) -> bool:
    """Restore the baseline prompts; False only if the agent or its baseline is missing."""
    agent_dir = find_agent_dir(agent_name, directory_provider=directory_provider)
    if not agent_dir:
        print(f"ERROR: Agent '{agent_name}' not found.")
        return False

    prompts_dir = agent_dir / "prompts"
    original_dir = prompts_dir / "baseline"

    if not original_dir.is_dir():
        print(f"ERROR: No baseline/ directory found at {original_dir}")
        print("This agent does not have stored baseline prompts to reset to.")
        return False

    original_files = sorted(original_dir.glob("*.md"))
    if not original_files:
        print(f"ERROR: No .md files found in {original_dir}")
        return False

    print(f"Agent:    {agent_name}")
    print(f"Prompts:  {prompts_dir}")
    print(f"Baseline: {original_dir}")
    print(f"Files:    {len(original_files)}")
    print()

    any_changed = False
    for orig_file in original_files:
        target = prompts_dir / orig_file.name
        orig_content = orig_file.read_text()
        target_content = target.read_text() if target.exists() else None

        if target_content == orig_content:
            print(f"  {orig_file.name}: already at original (no change)")
            continue

        old_size = len(target_content) if target_content is not None else 0
        new_size = len(orig_content)

        if dry_run:
            print(f"  {orig_file.name}: WOULD reset ({old_size} → {new_size} bytes)")
        else:
            shutil.copy2(orig_file, target)
            print(f"  {orig_file.name}: RESET ({old_size} → {new_size} bytes)")
        any_changed = True

    if not any_changed:
        print("\nAll prompts already at original — nothing to do.")
    elif dry_run:
        print("\nDry run — no files changed. Run without --dry-run to apply.")
    else:
        print("\nDone. All prompts reset to original baseline.")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset an agent's prompts to original (baseline) versions.",
    )
    parser.add_argument("agent_name", help="Agent to reset (e.g. 'claim_history')")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without actually writing files",
    )
    args = parser.parse_args()

    if not reset_prompts(args.agent_name, dry_run=args.dry_run):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
