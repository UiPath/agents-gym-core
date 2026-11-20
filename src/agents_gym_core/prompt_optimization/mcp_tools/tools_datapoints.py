"""MCP tools for eval-set datapoints and evaluators (mcp_server exposes only the read-only ones)."""

import json
import subprocess
import sys
from pathlib import Path

from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    directory_provider,
    find_datapoint,
    get_eval_set_and_datapoint,
    hidden_eval_sets,
    is_hidden_eval_set,
    list_eval_set_files,
    list_evaluator_files,
    load_eval_set,
    load_json_safe,
    optimization_target,
    repo_root,
    require_agent_dir,
    save_eval_set,
)


def check_evals(agent_name: str = "", source: str = "all") -> str:
    """Run the eval-set and evaluator checker on an agent or all agents.

    When source="all", runs the checker for both gym and core and combines results.
    When source="gym" or "core", runs only for that source. In an optimization run it checks
    only the target agent, like the other agent tools.
    """
    target = optimization_target()
    if target is not None and not agent_name:
        agent_name = target[0]
    excluded: list[str] = []  # in an optimization run, the splits the optimizer may not read
    if agent_name:
        agent_dir, err = require_agent_dir(agent_name, source)
        if err:
            return err
        agent_name = agent_dir.name  # the checker gets the plain folder name, never the raw string
        if target is not None:
            hidden = hidden_eval_sets(agent_dir)
            excluded = sorted(path.name for path in hidden if path.parent.name == "eval-sets")
    provider = directory_provider()
    sources = (
        ["core"]
        if isinstance(provider, ProjectDirectoryProvider)
        else (["gym", "core"] if source == "all" else [source])
    )
    all_output = []

    for src in sources:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "check_eval_sets.py"),
            "--check-evaluators",
        ]
        if isinstance(provider, ProjectDirectoryProvider):
            cmd += ["--agents-root", str(provider.get_agents_dir())]
        if agent_name:
            cmd.extend(["--agent", agent_name])
        for name in excluded:
            cmd.extend(["--exclude", name])
        if src == "core":
            cmd.append("--core")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(repo_root())
            )
            output = result.stdout
            if result.stderr:
                output += f"\n\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n\nExit code: {result.returncode} (errors found)"
            all_output.append(f"=== {src} ===\n{output or 'No output (check completed silently)'}")
        except subprocess.TimeoutExpired:
            all_output.append(f"=== {src} ===\nChecker timed out after 120s")
        except Exception as e:
            all_output.append(f"=== {src} ===\nError running checker: {e}")

    return "\n\n".join(all_output)


def get_datapoint(
    agent_name: str,
    datapoint_name: str,
    source: str = "all",
) -> str:
    """Get a specific datapoint from an eval-set by name. Searches all eval sets."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    eval_set_files = list_eval_set_files(agent_dir)
    if not eval_set_files:
        return f"No eval sets found for agent '{agent_name}'."

    all_names: list[str] = []
    seen_names: set[str] = set()
    for esf in eval_set_files:
        if is_hidden_eval_set(esf, agent_dir):
            continue
        data, err = load_eval_set(esf)
        if err:
            continue
        ev, _ = find_datapoint(data.get("evaluations", []), datapoint_name)
        if ev is not None:
            return json.dumps(ev, indent=2, default=str)
        for e in data.get("evaluations", []):
            n = e.get("name", "")
            if n and n not in seen_names:
                all_names.append(n)
                seen_names.add(n)

    names_str = "\n".join(f"  - {n}" for n in all_names)
    return f"Datapoint '{datapoint_name}' not found. Available ({len(all_names)}):\n{names_str}"


def update_datapoint_inputs(
    agent_name: str,
    datapoint_name: str,
    inputs_json: str,
    source: str = "all",
    eval_set_name: str = "default",
) -> str:
    """Update the inputs of a specific datapoint in an eval-set."""
    try:
        new_inputs = json.loads(inputs_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON for inputs: {e}"

    esf, data, ev, err = get_eval_set_and_datapoint(
        agent_name, datapoint_name, source, eval_set_name
    )
    if err:
        return err
    assert esf is not None and data is not None and ev is not None

    old_inputs = ev.get("inputs", {})
    ev["inputs"] = new_inputs
    save_eval_set(esf, data)
    return (
        f"Updated inputs for '{datapoint_name}'.\n"
        f"Old: {json.dumps(old_inputs)}\n"
        f"New: {json.dumps(new_inputs)}"
    )


def update_evaluation_criterias(
    agent_name: str,
    datapoint_name: str,
    evaluator_id: str,
    expected_output_json: str,
    expected_agent_behavior: str = "",
    source: str = "all",
    eval_set_name: str = "default",
) -> str:
    """Update the evaluationCriterias for a specific datapoint and evaluator."""
    try:
        new_eo = json.loads(expected_output_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON for expectedOutput: {e}"

    esf, data, ev, err = get_eval_set_and_datapoint(
        agent_name, datapoint_name, source, eval_set_name
    )
    if err:
        return err
    assert esf is not None and data is not None and ev is not None

    criterias = ev.get("evaluationCriterias", {})
    old_criteria = criterias.get(evaluator_id, {})
    criterias[evaluator_id] = {
        "expectedOutput": new_eo,
        "expectedAgentBehavior": expected_agent_behavior,
    }
    ev["evaluationCriterias"] = criterias
    save_eval_set(esf, data)
    return (
        f"Updated evaluationCriterias['{evaluator_id}'] for '{datapoint_name}'.\n"
        f"Old: {json.dumps(old_criteria, default=str)}\n"
        f"New: {json.dumps(criterias[evaluator_id])}"
    )


# ---------------------------------------------------------------------------
# Evaluator prompt editing
# ---------------------------------------------------------------------------


def update_evaluator_prompt(
    agent_name: str,
    evaluator_id: str,
    new_prompt: str,
    source: str = "all",
) -> str:
    """Update the prompt of an LLM judge evaluator (output judge or trajectory evaluator).

    Args:
        agent_name: Name of the agent
        evaluator_id: Evaluator ID (e.g. "myagent_LLMJudgeOutputEvaluator")
        new_prompt: The new evaluator prompt (full replacement)
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    evaluator_files = list_evaluator_files(agent_dir)
    if not evaluator_files:
        return f"No evaluators directory found for agent '{agent_name}'."

    # Find the evaluator file by ID
    evaluator_path: Path | None = None
    for ef in evaluator_files:
        data, load_err = load_json_safe(ef)
        if load_err or not isinstance(data, dict):
            continue
        if data.get("id") == evaluator_id:
            evaluator_path = ef
            break

    if evaluator_path is None:
        # List available evaluators to help
        available = []
        for ef in evaluator_files:
            data, load_err = load_json_safe(ef)
            if load_err or not isinstance(data, dict):
                available.append(f"  - {ef.name}: (error reading)")
                continue
            eid = data.get("id", "?")
            etype = data.get("evaluatorTypeId", "?")
            available.append(f"  - {eid} ({etype})")
        return (
            f"Evaluator '{evaluator_id}' not found for agent '{agent_name}'.\n\n"
            f"Available evaluators:\n" + "\n".join(available)
        )

    # Load the evaluator JSON
    eval_data, load_err = load_json_safe(evaluator_path)
    if load_err or not isinstance(eval_data, dict):
        return f"Error reading evaluator file: {load_err}"

    eval_config = eval_data.get("evaluatorConfig", {})
    if not isinstance(eval_config, dict):
        return f"Evaluator '{evaluator_id}' has no evaluatorConfig section."

    old_prompt = eval_config.get("prompt", "")
    if not old_prompt:
        return f"Evaluator '{evaluator_id}' has no prompt field in evaluatorConfig."

    # Validate that required placeholders are preserved
    eval_type = eval_data.get("evaluatorTypeId", "")
    if "output" in eval_type.lower():
        for placeholder in ("{{ExpectedOutput}}", "{{ActualOutput}}"):
            if placeholder not in new_prompt:
                return (
                    f"Output judge evaluator requires {placeholder} placeholder in the prompt. "
                    f"Please include it."
                )
    elif "trajectory" in eval_type.lower():
        for placeholder in ("{{ExpectedAgentBehavior}}", "{{AgentRunHistory}}"):
            if placeholder not in new_prompt:
                return (
                    f"Trajectory evaluator requires {placeholder} placeholder in the prompt. "
                    f"Please include it."
                )

    # Update and save
    eval_config["prompt"] = new_prompt
    try:
        evaluator_path.write_text(json.dumps(eval_data, indent=2, ensure_ascii=False) + "\n")
    except (OSError, UnicodeError):
        return "Error writing the selected evaluator."

    return (
        f"Updated evaluator prompt for '{evaluator_id}'.\n"
        f"File: {evaluator_path.relative_to(agent_dir.resolve())}\n"
        f"Type: {eval_type}\n"
        f"Old length: {len(old_prompt)} chars\n"
        f"New length: {len(new_prompt)} chars"
    )
