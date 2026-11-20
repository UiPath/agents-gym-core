"""MCP tools for listing and inspecting agents."""

import json
import os
import stat
import tempfile
from pathlib import Path

import yaml

from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    _resolve_agent_path,
    _valid_relative_path,
    extract_tool_names,
    format_size,
    format_tool_schema,
    is_hidden_eval_set,
    is_hidden_prompt,
    list_agent_files,
    list_eval_set_files,
    list_evaluator_files,
    load_agent_config,
    require_agent_dir,
    resolve_agent_directory,
    resolve_agent_file,
    resolve_prompt_path,
    summarize_eval_set_file,
)
from agents_gym_core.prompt_optimization.mcp_tools.usage_tracker import record_prompt_update


def get_agent_files(agent_name: str, source: str = "all") -> str:
    """List all files in an agent's directory tree with relative paths and sizes."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    lines = [f"=== Files for {agent_name} ({source}) ==="]
    total_size = 0
    file_count = 0

    for f in list_agent_files(agent_dir):
        contained = resolve_agent_file(agent_dir, str(f.relative_to(agent_dir)))
        if contained is None:
            continue
        try:
            size = contained.stat().st_size
        except OSError:
            continue
        total_size += size
        file_count += 1
        lines.append(f"  {f.relative_to(agent_dir)}  ({format_size(size)})")

    lines.insert(1, f"Total: {file_count} files, {format_size(total_size)}\n")
    return "\n".join(lines)


def read_agent_file(agent_name: str, relative_path: str, source: str = "all") -> str:
    """Read a file from an agent's directory by relative path."""
    if not _valid_relative_path(relative_path):
        return "Invalid relative_path: expected a path within the selected agent."

    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    file_path = _resolve_agent_path(agent_dir, relative_path)
    if file_path is None:
        return "Refused: resolved path is not within the selected agent."
    if is_hidden_eval_set(file_path, agent_dir):
        return f"Access denied: '{relative_path}' is not in this optimization run's split."
    if is_hidden_prompt(file_path, agent_dir):
        return f"Access denied: '{relative_path}' belongs to another version of the agent."
    try:
        exists = file_path.exists()
    except OSError:
        return "Error accessing the selected agent file."
    if not exists:
        # List available files to help
        available = [str(f.relative_to(agent_dir)) for f in list_agent_files(agent_dir)]
        return (
            f"File not found: {relative_path}\n\n"
            f"Available files ({len(available)}):\n" + "\n".join(f"  {f}" for f in available)
        )

    file_path = resolve_agent_file(agent_dir, relative_path)
    if file_path is None:
        return f"Not a regular file: {relative_path}"

    # Safety: don't read huge files
    try:
        size = file_path.stat().st_size
    except OSError:
        return "Error accessing the selected agent file."
    if size > 500_000:
        return f"File too large to read: {relative_path} ({format_size(size)}). Max 500KB."

    try:
        content = file_path.read_text()
    except UnicodeDecodeError:
        return f"Cannot read binary file: {relative_path} ({format_size(size)})"
    except Exception:
        return f"Error reading {relative_path}."

    return f"=== {agent_name}/{relative_path} ({format_size(size)}) ===\n\n{content}"


def summarize_agent(agent_name: str, source: str = "all") -> str:
    """Get a concise summary of an agent."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    config, err = load_agent_config(agent_dir)
    if err:
        return err

    meta = config.get("metadata", {})
    agent_cfg = config.get("agent", {})

    lines = [f"# {agent_name}"]
    lines.append(f"**{meta.get('description', 'No description')}**")
    lines.append(f"Tags: {meta.get('tags', [])}")

    tools = agent_cfg.get("tools", [])
    lines.append(f"\nTools ({len(tools)}): {', '.join(extract_tool_names(tools))}")

    input_fields = list(agent_cfg.get("input_schema", {}).get("fields", {}).keys())
    output_fields = list(agent_cfg.get("output_schema", {}).get("fields", {}).keys())
    lines.append(f"Inputs: {input_fields}")
    lines.append(f"Outputs: {output_fields}")

    for esf in list_eval_set_files(agent_dir):
        if not is_hidden_eval_set(esf, agent_dir):
            lines.append(f"\nEval-set '{esf.stem}': {summarize_eval_set_file(esf)}")

    sp_ref = agent_cfg.get("system_prompt", "")
    if sp_ref:
        sp_path = resolve_prompt_path(agent_dir, sp_ref)
        if sp_path is not None and not is_hidden_eval_set(sp_path, agent_dir):
            try:
                sp_text = sp_path.read_text()
            except (OSError, UnicodeError):
                sp_text = ""
            snippet = sp_text[:300].strip()
            if len(sp_text) > 300:
                snippet += "..."
            lines.append(f"\nSystem prompt preview:\n{snippet}")

    return "\n".join(lines)


def inspect_agent(agent_name: str, source: str = "all") -> str:
    """Deep-inspect an agent's configuration."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    config, err = load_agent_config(agent_dir)
    if err:
        return err

    lines = [f"=== Agent: {agent_name} ({source}) ==="]

    meta = config.get("metadata", {})
    if meta:
        lines.append(f"Description: {meta.get('description', 'N/A')}")
        lines.append(f"Tags: {meta.get('tags', [])}")

    init_cfg = config.get("init", {})
    data_dirs = init_cfg.get("data_dirs", []) if isinstance(init_cfg, dict) else []
    if data_dirs:
        lines.append(f"Init data_dirs: {data_dirs}")

    agent_cfg = config.get("agent", {})

    tools = agent_cfg.get("tools", [])
    lines.append(f"\nTools ({len(tools)}):")
    for t in tools:
        target = t.get("_target_", "unknown") if isinstance(t, dict) else str(t)
        lines.append(f"  - {target}")

    input_schema = agent_cfg.get("input_schema", {}).get("fields", {})
    output_schema = agent_cfg.get("output_schema", {}).get("fields", {})
    lines.append(f"\nInput fields ({len(input_schema)}):")
    for fname, fdef in input_schema.items():
        lines.append(f"  {fname}: {fdef.get('type', '?')} - {fdef.get('description', '')[:100]}")
    lines.append(f"\nOutput fields ({len(output_schema)}):")
    for fname, fdef in output_schema.items():
        lines.append(f"  {fname}: {fdef.get('type', '?')} - {fdef.get('description', '')[:100]}")

    weights = agent_cfg.get("evaluator_weights", {})
    if weights:
        lines.append("\nEvaluator weights:")
        for ew, wv in weights.items():
            lines.append(f"  {ew}: {wv}")

    eval_sets = agent_cfg.get("evaluation_sets", [])
    lines.append(f"\nEvaluation sets referenced: {len(eval_sets)}")
    for es in eval_sets:
        lines.append(f"  - {es}")

    for esf in list_eval_set_files(agent_dir):
        if not is_hidden_eval_set(esf, agent_dir):
            lines.append(f"\n  {summarize_eval_set_file(esf)}")

    evaluator_files = list_evaluator_files(agent_dir)
    if evaluator_files:
        lines.append(f"\nEvaluator files: {len(evaluator_files)}")
        for ef in evaluator_files:
            try:
                edata = json.loads(ef.read_text())
                lines.append(
                    f"  - {ef.name}: id={edata.get('id', '?')}, type={edata.get('evaluatorTypeId', '?')}"
                )
            except Exception:
                lines.append(f"  - {ef.name}: (error reading)")

    return "\n".join(lines)


def inspect_tools(agent_name: str, source: str = "all") -> str:
    """Inspect an agent's tools in detail."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    config, err = load_agent_config(agent_dir)
    if err:
        return err

    tools = config.get("agent", {}).get("tools", [])
    if not tools:
        return f"Agent '{agent_name}' has no tools configured."

    lines = [f"=== Tools for {agent_name} ({len(tools)} tools) ==="]

    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            lines.append(f"\n[{i}] {t}")
            continue

        target = t.get("_target_", "unknown")
        lines.append(f"\n[{i}] {target}")

        for k, v in t.items():
            if k == "_target_":
                continue
            if k == "tool_schema" and isinstance(v, dict):
                lines.extend(format_tool_schema(v))
            else:
                lines.append(f"  {k}: {v}")

        tool_name = t.get("tool_name", "")
        if isinstance(tool_name, str) and tool_name:
            data_file = resolve_agent_file(
                agent_dir, str(Path("tools_data") / tool_name / "data.json")
            )
            if data_file is not None and not is_hidden_eval_set(data_file, agent_dir):
                try:
                    td = json.loads(data_file.read_text())
                    if isinstance(td, dict):
                        lines.append(f"  tracedump: {len(td)} entries in data.json")
                        for sk in list(td.keys())[:3]:
                            lines.append(f"    key: {sk[:80]}")
                    elif isinstance(td, list):
                        lines.append(f"  tracedump: {len(td)} entries in data.json")
                except Exception:
                    lines.append("  tracedump: data.json (error reading)")

    tools_data_dir = resolve_agent_directory(agent_dir, "tools_data")
    if tools_data_dir is not None:
        yaml_tool_names = {t.get("tool_name", "") for t in tools if isinstance(t, dict)}
        try:
            tool_directories = sorted(tools_data_dir.iterdir())
        except OSError:
            tool_directories = []
        for td_dir in tool_directories:
            if td_dir.name in yaml_tool_names:
                continue
            data_file = resolve_agent_file(
                agent_dir, str((td_dir / "data.json").relative_to(agent_dir.resolve()))
            )
            if data_file is not None and not is_hidden_eval_set(data_file, agent_dir):
                lines.append(
                    f"\n[extra] tools_data/{td_dir.name}/data.json exists but not in agent.yaml tools"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt read/update
# ---------------------------------------------------------------------------


def _resolve_prompt(
    agent_name: str, prompt_type: str, source: str
) -> tuple[str | None, str | None, str | None]:
    """Resolve a prompt file path. Returns (file_path, content, None) or (None, None, error)."""
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return None, None, err

    config, err = load_agent_config(agent_dir)
    if err:
        return None, None, err

    # Standard prompts: "system" → agent.system_prompt, "user" → agent.user_prompt.
    # Signature prompts: any other value → agent.signature_prompts.<prompt_type>.
    if prompt_type == "system":
        key = "system_prompt"
        prompt_ref = config.get("agent", {}).get(key, "")
    elif prompt_type == "user":
        key = "user_prompt"
        prompt_ref = config.get("agent", {}).get(key, "")
    else:
        key = f"signature_prompts.{prompt_type}"
        prompt_ref = config.get("agent", {}).get("signature_prompts", {}).get(prompt_type, "")

    if not prompt_ref:
        return None, None, f"No {key} configured in agent.yaml for '{agent_name}'."

    prompt_path = resolve_prompt_path(agent_dir, prompt_ref)
    if not prompt_path:
        return None, None, "Prompt file not found within the selected agent."
    if is_hidden_eval_set(prompt_path, agent_dir):
        return None, None, "Access denied: prompt is not in this optimization run's split."

    try:
        return str(prompt_path), prompt_path.read_text(), None
    except (OSError, UnicodeError):
        return None, None, "Error reading the selected prompt."


def list_prompt_types(agent_name: str, source: str = "all") -> list[str]:
    """Discover all available prompt types for an agent from agent.yaml.

    Returns a list of prompt_type strings (e.g. ["system", "user", "summarize1"]).
    Only includes prompts that are configured in agent.yaml.
    """
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return ["system"]

    config, err = load_agent_config(agent_dir)
    if err:
        return ["system"]

    agent_cfg = config.get("agent", {})
    prompt_types: list[str] = []

    if agent_cfg.get("system_prompt"):
        prompt_types.append("system")
    if agent_cfg.get("user_prompt"):
        prompt_types.append("user")

    sig_prompts = agent_cfg.get("signature_prompts", {})
    if isinstance(sig_prompts, dict):
        for name in sig_prompts:
            prompt_types.append(name)

    return prompt_types or ["system"]


def get_prompt(agent_name: str, prompt_type: str = "system", source: str = "all") -> str:
    """Read any configured agent prompt. Returns the raw prompt content."""
    _path, content, err = _resolve_prompt(agent_name, prompt_type, source)
    if err:
        return err
    assert content is not None
    return content


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temp file + os.replace."""
    original_mode = path.stat().st_mode if path.exists() else None
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(content)
        if original_mode is not None:
            os.chmod(tmp_path, stat.S_IMODE(original_mode))
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def update_prompt(
    agent_name: str,
    new_content: str,
    prompt_type: str = "system",
    source: str = "all",
    round_name: str = "",
) -> str:
    """Replace any configured agent prompt."""
    path, old_content, err = _resolve_prompt(agent_name, prompt_type, source)
    if err:
        return err
    assert path is not None and old_content is not None

    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err
    try:
        relative_path = str(Path(path).resolve().relative_to(agent_dir.resolve()))
    except (OSError, RuntimeError, ValueError):
        return "Refused: prompt path is not within the selected agent."
    prompt_path = resolve_agent_file(agent_dir, relative_path)
    if prompt_path is None:
        return "Refused: prompt path is not within the selected agent."
    if is_hidden_eval_set(prompt_path, agent_dir):
        return "Access denied: prompt is not in this optimization run's split."

    try:
        _atomic_write(prompt_path, new_content)
    except (OSError, UnicodeError):
        return "Error writing the selected prompt."

    # Track usage
    record_prompt_update(agent_name, prompt_type, round_name)

    return (
        f"Updated {prompt_type} prompt for '{agent_name}'.\n"
        f"File: {relative_path}\n"
        f"Old length: {len(old_content)} chars\n"
        f"New length: {len(new_content)} chars"
    )


# ---------------------------------------------------------------------------
# agent.yaml update
# ---------------------------------------------------------------------------


def update_agent_yaml(
    agent_name: str,
    yaml_path: str,
    value_json: str,
    source: str = "all",
) -> str:
    """Update a specific field in agent.yaml using dot-notation path.

    Args:
        agent_name: Name of the agent
        yaml_path: Dot-notation path to the field (e.g. "metadata.tags", "agent.output_schema.fields.Status.description")
        value_json: New value as a JSON string (e.g. '"new description"', '["tag1","tag2"]', '{"type":"str","description":"..."}')
        source: "all" (default) to search both gym and core, or "gym"/"core" for a specific source
    """
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return err

    config_file = resolve_agent_file(agent_dir, "agent.yaml")
    if config_file is None:
        return f"agent.yaml not found for agent '{agent_name}'."

    try:
        new_value = json.loads(value_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON for value: {e}\nProvided: {value_json}"

    config, err = load_agent_config(agent_dir)
    if err:
        return err

    # Navigate the dot-path
    keys = yaml_path.split(".")
    if not keys:
        return "yaml_path cannot be empty."

    # Traverse to parent, keeping track for error messages
    current = config
    for i, key in enumerate(keys[:-1]):
        if not isinstance(current, dict):
            path_so_far = ".".join(keys[:i])
            return f"Cannot traverse '{yaml_path}': '{path_so_far}' is not a mapping (it's {type(current).__name__})."
        if key not in current:
            path_so_far = ".".join(keys[: i + 1])
            available = list(current.keys())
            return f"Key '{path_so_far}' not found in agent.yaml. Available keys at this level: {available}"
        current = current[key]

    # Set the final key
    final_key = keys[-1]
    if not isinstance(current, dict):
        path_so_far = ".".join(keys[:-1])
        return f"Cannot set '{final_key}' on '{path_so_far}': it's {type(current).__name__}, not a mapping."

    old_value = current.get(final_key, "<not set>")
    current[final_key] = new_value

    # Write back
    yaml_out = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    assert isinstance(yaml_out, str)
    try:
        _atomic_write(config_file, yaml_out)
    except (OSError, UnicodeError):
        return "Error writing agent.yaml."

    return (
        f"Updated agent.yaml for '{agent_name}'.\n"
        f"Path: {yaml_path}\n"
        f"Old: {json.dumps(old_value, default=str)}\n"
        f"New: {json.dumps(new_value, default=str)}"
    )
