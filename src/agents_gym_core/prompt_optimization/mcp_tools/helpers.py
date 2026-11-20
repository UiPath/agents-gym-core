"""Shared helpers for the optimizer's MCP server and the prompt optimizers."""

import io
import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agents_gym_core.utils.directory_provider import CoreDirectoryProvider, DirectoryProvider
from agents_gym_core.utils.gym_settings_core import gym_settings_core

DIRECTORY_PROVIDER: DirectoryProvider = CoreDirectoryProvider()


_PROVIDER: ContextVar[DirectoryProvider | None] = ContextVar(
    "optimizer_directory_provider", default=None
)


@contextmanager
def optimization_context(provider: DirectoryProvider | None):
    """Bind one optimizer/server invocation; nested scopes restore their caller."""
    token = _PROVIDER.set(provider)
    try:
        yield
    finally:
        _PROVIDER.reset(token)


def directory_provider() -> DirectoryProvider:
    return _PROVIDER.get() or DIRECTORY_PROVIDER


def repo_root() -> Path:
    return directory_provider().get_project_root()


def experiments_dir() -> Path:
    return repo_root() / "experiments"


# ---------------------------------------------------------------------------
# MCP defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_MAX_DATAPOINTS = 0  # 0 = full eval-set
DEFAULT_EXPERIMENT_NAME = "mcp_run"


# ---------------------------------------------------------------------------
# Agent directory helpers
# ---------------------------------------------------------------------------


AGENT_NAME = re.compile(r"[A-Za-z0-9_-]+")


def resolve_agent_dir(agent_name: str, source: str) -> Path | None:
    """Return the agent directory or None if not found.

    Accepts version-qualified names (e.g. "my_agent:v1") — the version suffix
    is stripped before the filesystem lookup since versions live in versions.yaml,
    not in separate directories.

    When source="all", searches gym first, then core. Names that are not plain folder names
    (e.g. containing "/" or "..") resolve to nothing.
    """
    if not isinstance(agent_name, str) or any(c in agent_name for c in ("/", "\\", "\x00")):
        return None
    base_name = agent_name.split(":")[0]
    if not AGENT_NAME.fullmatch(base_name):
        return None
    roots = directory_provider().agent_roots(source)
    for root in roots:
        agent_dir = root / base_name
        try:
            resolved_root = root.resolve()
            resolved_agent = agent_dir.resolve()
            if resolved_agent == resolved_root or not resolved_agent.is_relative_to(resolved_root):
                continue
            if agent_dir.is_dir() and resolve_agent_file(agent_dir, "agent.yaml") is not None:
                return agent_dir
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def require_agent_dir(agent_name: str, source: str) -> tuple[Path, str | None]:
    """Return (agent_dir, None) or (_, error_message).

    In an optimization run only the target agent resolves: another agent's files and prompts
    (e.g. a related benchmark's released prompts) are neither readable nor writable.
    """
    agent_dir = resolve_agent_dir(agent_name, source)
    if not agent_dir:
        return Path(), f"Agent '{agent_name}' not found in {source} agents."
    target = optimization_target()
    if target is not None:
        target_dir = resolve_agent_dir(target[0], "all")
        if target_dir is None or not agent_dir.samefile(target_dir):
            return Path(), f"Access denied: this optimization run is limited to '{target[0]}'."
    return agent_dir, None


# ---------------------------------------------------------------------------
# Optimization target: the one agent version a run may evaluate and read
# ---------------------------------------------------------------------------

TARGET_ENV = "COACH_TARGET_AGENT"
# The optimization run's eval model (gym optimize -m): rounds are scored on it only.
EVAL_MODEL_ENV = "COACH_EVAL_MODEL"
# Its eval loops (gym optimize --loops; empty = the testbed default, basic), likewise.
EVAL_LOOPS_ENV = "COACH_EVAL_LOOPS"


def parse_spec(spec: str) -> tuple[str, str]:
    name, _, version = spec.strip().partition(":")
    return name, version or "base"


def optimization_target() -> tuple[str, str] | None:
    """(agent, version) the optimizer was started on, or None outside an optimization run."""
    spec = os.environ.get(TARGET_ENV, "").strip()
    return parse_spec(spec) if spec else None


def target_eval_sets(agent_dir: Path, version: str) -> set[Path]:
    """Eval-set files the target version evaluates on (its own, else the base agent's)."""
    config, err = load_agent_config(agent_dir)
    if err:
        return set()
    refs = (config.get("agent") or {}).get("evaluation_sets") or []
    if version != "base":
        versions_path = _resolve_agent_path(agent_dir, "versions.yaml")
        if versions_path is None:
            return set()
        try:
            if versions_path.exists():
                versions_file = resolve_agent_file(agent_dir, "versions.yaml")
                if versions_file is None:
                    return set()
                entry = (yaml.safe_load(versions_file.read_text()) or {}).get(version) or {}
                refs = (entry.get("agent") or {}).get("evaluation_sets", refs)
        except (OSError, ValueError, AttributeError, yaml.YAMLError):
            return set()
    if not isinstance(refs, list):
        return set()
    return {path for ref in refs if (path := resolve_prompt_path(agent_dir, ref)) is not None}


def hidden_eval_sets(agent_dir: Path) -> set[Path]:
    """Eval-set files of the optimization target that its pinned version does not use, and its
    mock_data directory."""
    target = optimization_target()
    target_dir = resolve_agent_dir(target[0], "all") if target else None
    # By identity: another spelling of the name (case on macOS/Windows) is the same folder.
    if target is None or target_dir is None or not agent_dir.samefile(target_dir):
        return set()
    mock_files: set[Path] = set()
    for name in directory_provider().private_data_dirs():
        data = resolve_agent_directory(agent_dir, name)
        if data:
            mock_files.update(f for f in data.rglob("*") if f.is_file())
    directory = _eval_sets_directory(agent_dir)
    if directory is None:
        return mock_files
    # Keep hiding aliases to other files inside the agent even when eval-set lookup
    # rejects them for leaving the narrower eval-set root.
    root = agent_dir.resolve()
    present = {
        path
        for candidate in directory.glob("*.json")
        if (path := resolve_agent_file(agent_dir, str(candidate.relative_to(root)))) is not None
    }
    return (present - target_eval_sets(agent_dir, target[1])) | mock_files


def is_hidden_eval_set(path: Path, agent_dir: Path) -> bool:
    """By file identity, so another spelling of the path (case, links) cannot reach it."""
    try:
        return path.is_file() and any(path.samefile(h) for h in hidden_eval_sets(agent_dir))
    except (OSError, RuntimeError, ValueError):
        return True


def is_hidden_prompt(path: Path, agent_dir: Path) -> bool:
    """Other versions' prompts (subfolders of prompts/) of the optimization target, by identity."""
    target = optimization_target()
    target_dir = resolve_agent_dir(target[0], "all") if target else None
    if target is None or target_dir is None or not agent_dir.samefile(target_dir):
        return False
    prompts = resolve_agent_directory(agent_dir, "prompts")
    if prompts is None:
        return False
    try:
        return path.is_file() and any(
            path.samefile(f) for f in prompts.glob("*/**/*") if f.is_file()
        )
    except (OSError, RuntimeError, ValueError):
        return True


def _workspace() -> Path | None:
    env_dir = os.environ.get("COACH_OUTPUT_DIR", "").strip()
    if not env_dir:
        return None
    return Path(env_dir) if Path(env_dir).is_absolute() else repo_root() / env_dir


# A hidden file: save_optimizer_log refuses hidden paths, so the optimizer cannot add to it.
STARTED_EXPERIMENTS_FILE = ".started_experiments.json"
EXPERIMENT_NAME = re.compile(r"[\w.:+-]+")


_REGISTRY_LOCK = threading.Lock()  # the MCP server runs sync tools in worker threads


def record_started_experiment(experiment_name: str) -> None:
    workspace = _workspace()
    if workspace is None:
        return
    workspace.mkdir(parents=True, exist_ok=True)
    with _REGISTRY_LOCK:
        names = sorted(started_experiments() | {experiment_name})
        tmp = workspace / f"{STARTED_EXPERIMENTS_FILE}.tmp"
        tmp.write_text(json.dumps(names, indent=1) + "\n")
        tmp.replace(workspace / STARTED_EXPERIMENTS_FILE)


def started_experiments() -> set[str]:
    """Experiments this optimization run started (kept in its workspace, so -c keeps them)."""
    workspace = _workspace()
    path = workspace / STARTED_EXPERIMENTS_FILE if workspace else None
    if path is None or not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def init_started_experiments(workspace: Path) -> None:
    """Create the registry before the optimizer can act. A workspace without one
    gets its rounds' experiments from ``round_NNN/round_stats.json``, read only this once."""
    path = workspace / STARTED_EXPERIMENTS_FILE
    if path.exists():
        return
    names = set()
    for stats in workspace.glob("round_*/round_stats.json"):
        data, err = load_json_safe(stats)
        if not err and isinstance(data, dict) and isinstance(data.get("experiment_name"), str):
            names.add(data["experiment_name"])
    workspace.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(names), indent=1) + "\n")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_agent_config(agent_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Load and return agent.yaml as a dict, or ({}, error_message)."""
    config_file = resolve_agent_file(agent_dir, "agent.yaml")
    if config_file is None:
        return {}, "agent.yaml not found within the selected agent."
    try:
        data = yaml.safe_load(config_file.read_text())
        if not isinstance(data, dict):
            return {}, "agent.yaml is not a valid YAML mapping."
        return data, None
    except Exception:
        return {}, "Error reading agent.yaml."


def _valid_relative_path(relative: str) -> bool:
    return (
        isinstance(relative, str)
        and bool(relative)
        and not any(c in relative for c in ("\\", ":", "\x00"))
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts
    )


def _resolve_agent_path(agent_dir: Path, relative: str) -> Path | None:
    """Resolve a relative path before checking its type or accessing its contents."""
    if not _valid_relative_path(relative):
        return None
    try:
        root = agent_dir.resolve()
        path = (root / relative).resolve()
        if path.is_relative_to(root):
            return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def resolve_agent_file(agent_dir: Path, relative: str) -> Path | None:
    """Resolve an existing regular file without leaving the selected agent."""
    path = _resolve_agent_path(agent_dir, relative)
    try:
        return path if path is not None and path.is_file() else None
    except (OSError, ValueError):
        return None


def resolve_agent_directory(agent_dir: Path, relative: str) -> Path | None:
    """Resolve a directory before enumerating it, including contained aliases."""
    path = _resolve_agent_path(agent_dir, relative)
    try:
        return path if path is not None and path.is_dir() else None
    except (OSError, ValueError):
        return None


def list_agent_files(agent_dir: Path) -> list[Path]:
    """List contained files, retaining their relative spelling for display."""
    try:
        return [
            candidate
            for candidate in sorted(agent_dir.rglob("*"))
            if not candidate.name.startswith(".")
            and "__pycache__" not in candidate.parts
            and resolve_agent_file(agent_dir, str(candidate.relative_to(agent_dir))) is not None
            and not is_hidden_eval_set(candidate, agent_dir)
            and not is_hidden_prompt(candidate, agent_dir)
        ]
    except (OSError, RuntimeError, ValueError):
        return []


def resolve_prompt_path(agent_dir: Path, prompt_ref: str) -> Path | None:
    """Resolve a namespaced or relative prompt file within the selected agent."""
    if not isinstance(prompt_ref, str):
        return None
    if "::" in prompt_ref:
        namespace, relative = prompt_ref.split("::", 1)
        if not AGENT_NAME.fullmatch(namespace):
            return None
    else:
        relative = prompt_ref
    return resolve_agent_file(agent_dir, relative)


def list_evaluator_files(agent_dir: Path) -> list[Path]:
    """List evaluator files without enumerating escaping directory aliases."""
    directory = resolve_agent_directory(agent_dir, "evaluations/evaluators")
    if directory is None:
        return []
    try:
        root = agent_dir.resolve()
        return [
            path
            for candidate in sorted(directory.glob("evaluator-*.json"))
            if (path := resolve_agent_file(agent_dir, str(candidate.relative_to(root)))) is not None
            and not is_hidden_eval_set(path, agent_dir)
        ]
    except (OSError, RuntimeError, ValueError):
        return []


def get_agent_tags(agent_dir: Path) -> list[str]:
    """Read tags from agent.yaml."""
    config, _ = load_agent_config(agent_dir)
    return config.get("metadata", {}).get("tags", []) if config else []


def load_json_safe(path: Path) -> tuple[object, str | None]:
    """Load a JSON file. Returns (data, None) or (None, error_message)."""
    try:
        return json.loads(path.read_text()), None
    except Exception:
        return None, f"Error reading {path.name}."


# ---------------------------------------------------------------------------
# Eval-set helpers
# ---------------------------------------------------------------------------


def _eval_sets_directory(agent_dir: Path) -> Path | None:
    return resolve_agent_directory(agent_dir, "evaluations/eval-sets")


def _contained_eval_set(directory: Path, candidate: Path) -> Path | None:
    try:
        path = candidate.resolve()
        if path.is_relative_to(directory) and path.is_file():
            return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def list_eval_set_files(agent_dir: Path) -> list[Path]:
    """List regular eval sets contained within the selected agent's eval-set root."""
    directory = _eval_sets_directory(agent_dir)
    if directory is None:
        return []
    try:
        return [
            path
            for candidate in sorted(directory.glob("evaluation-set-*.json"))
            if (path := _contained_eval_set(directory, candidate)) is not None
        ]
    except (OSError, RuntimeError, ValueError):
        return []


def find_eval_set_file(agent_dir: Path, eval_set_name: str = "default") -> Path | None:
    """Find a contained eval-set by exact suffix, then a partial-name match."""
    if (
        not _valid_relative_path(eval_set_name)
        or "/" in eval_set_name
        or eval_set_name in (".", "..")
    ):
        return None
    eval_sets_dir = _eval_sets_directory(agent_dir)
    if eval_sets_dir is None:
        return None

    target = f"evaluation-set-{eval_set_name}.json"
    path = _contained_eval_set(eval_sets_dir, eval_sets_dir / target)
    if path is not None:
        return path
    try:
        for candidate in sorted(eval_sets_dir.glob("evaluation-set-*.json")):
            if eval_set_name in candidate.name:
                path = _contained_eval_set(eval_sets_dir, candidate)
                if path is not None:
                    return path
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def load_eval_set(path: Path) -> tuple[dict, str | None]:
    """Load an eval-set JSON file. Returns (data, None) or ({}, error_message)."""
    try:
        return json.loads(path.read_text()), None
    except Exception:
        return {}, f"Error reading {path.name}."


def save_eval_set(path: Path, data: dict) -> None:
    """Write an eval-set JSON back to disk."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def find_datapoint(evaluations: list[dict], datapoint_name: str) -> tuple[dict | None, str | None]:
    """Find a datapoint by exact or partial name match.

    Returns (datapoint, None) or (None, error_message_with_available_names).
    """
    for ev in evaluations:
        if ev.get("name") == datapoint_name:
            return ev, None
    for ev in evaluations:
        if datapoint_name.lower() in ev.get("name", "").lower():
            return ev, None
    names = [ev.get("name", "?") for ev in evaluations]
    return None, (
        f"Datapoint '{datapoint_name}' not found. Available ({len(names)}):\n"
        + "\n".join(f"  - {n}" for n in names)
    )


def get_eval_set_and_datapoint(
    agent_name: str, datapoint_name: str, source: str, eval_set_name: str
) -> tuple[Path | None, dict | None, dict | None, str | None]:
    """Locate agent dir, load eval-set, find datapoint.

    Returns (eval_set_path, eval_set_data, datapoint, None) on success,
    or (None, None, None, error_message) on failure.
    """
    agent_dir, err = require_agent_dir(agent_name, source)
    if err:
        return None, None, None, err

    esf = find_eval_set_file(agent_dir, eval_set_name)
    if not esf:
        return None, None, None, f"Eval-set '{eval_set_name}' not found for agent '{agent_name}'."
    if is_hidden_eval_set(esf, agent_dir):
        return None, None, None, "Access denied: eval-set is not in this optimization run's split."

    data, err = load_eval_set(esf)
    if err:
        return None, None, None, err

    ev, err = find_datapoint(data.get("evaluations", []), datapoint_name)
    if err:
        return None, None, None, err

    return esf, data, ev, None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_size(size_bytes: int) -> str:
    """Format a file size in bytes to a human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def format_score(score: object) -> str:
    """Format a score value as a percentage string."""
    if isinstance(score, (int, float)):
        return f"{score * 100:.1f}%"
    return str(score)


def summarize_eval_set_file(esf: Path) -> str:
    """Return a one-line summary of an eval-set file."""
    try:
        data = json.loads(esf.read_text())
        n = len(data.get("evaluations", []))
        refs = data.get("evaluatorRefs", [])
        return f"{esf.name}: {n} datapoints, evaluators={refs}"
    except Exception:
        return f"{esf.name}: (error reading)"


def extract_tool_names(tools: list) -> list[str]:
    """Extract short tool names from agent.yaml tools list."""
    names = []
    for t in tools:
        if isinstance(t, dict):
            target = t.get("_target_", "")
            names.append(target.rsplit(".", 1)[-1] if "." in target else target)
        else:
            names.append(str(t))
    return names


def format_tool_schema(schema: dict) -> list[str]:
    """Format a tool_schema dict into display lines."""
    lines = [
        "  tool_schema:",
        f"    name: {schema.get('name', '?')}",
        f"    description: {schema.get('description', '?')[:100]}",
    ]
    args_schema = schema.get("args_schema", {})
    props = args_schema.get("properties", {})
    required = args_schema.get("required", [])
    lines.append(f"    args ({len(props)}):")
    for pname, pval in props.items():
        req = " [required]" if pname in required else ""
        ptype = pval.get("type", "?")
        pdesc = pval.get("description", "")[:80]
        lines.append(f"      {pname}: {ptype}{req} - {pdesc}")
    return lines


def format_experiment_scores(scores: dict) -> list[str]:
    """Format final_score_per_agent.json into display lines."""
    lines: list[str] = []
    for loop_key, agents in scores.items():
        if not isinstance(agents, dict):
            continue
        for agent_key, d in agents.items():
            if not isinstance(d, dict):
                continue
            all_data = d.get("all_data", {})
            mean = format_score(all_data.get("runs_mean", "N/A"))
            ci = all_data.get("runs_ci", [])
            runs = all_data.get("runs", {})
            ci_str = ""
            if isinstance(ci, list) and len(ci) == 2:
                ci_str = f" CI=[{float(ci[0]) * 100:.1f}%, {float(ci[1]) * 100:.1f}%]"
            lines.append(f"    {agent_key} [{loop_key}]: {mean}{ci_str} ({len(runs)} runs)")
    return lines


# ---------------------------------------------------------------------------
# Optimizer shared utilities
# ---------------------------------------------------------------------------


class TeeWriter(io.TextIOBase):
    """Write to both the original stream and a log file."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase):
        self._original = original
        self._log_file = log_file

    def write(self, s: str) -> int:
        self._original.write(s)
        self._log_file.write(s)
        self._log_file.flush()
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()


@contextmanager
def tee_to_log(output_dir: str, prefix: str):
    """Context manager that tees stdout to a log file. Yields the log file path."""
    output_path = repo_root() / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = output_path / f"{prefix}_session_{timestamp}.log"
    log_fh = open(log_file_path, "w")
    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_fh)  # type: ignore
    try:
        yield log_file_path
    finally:
        sys.stdout = original_stdout
        log_fh.close()


def compute_output_dir(
    agent_spec: str,
    optimizer_model: str,
    model: str,
    name: str,
    skill: str = "",
) -> str:
    agent_name = agent_spec.split(":")[0]
    agent_version = agent_spec.split(":")[1] if ":" in agent_spec else None
    # Sanitize only filesystem-illegal chars (':', '/'); dots are fine in dir names.
    opt_safe = re.sub(r"[:/]+", "_", optimizer_model)
    loop_safe = re.sub(r"[:/]+", "_", model)
    skill_suffix = f"_skill_{skill}" if skill else ""
    temp = str(gym_settings_core.UIPATH_AGENTS_GYM_TEMPERATURE)
    temp_suffix = f"_temp_{temp}"
    if agent_version:
        return f"tmp_{name}_{agent_name}_{agent_version}_optimizer_{opt_safe}_loopmodel_{loop_safe}{skill_suffix}{temp_suffix}"
    return f"tmp_{name}_{agent_name}_optimizer_{opt_safe}_loopmodel_{loop_safe}{skill_suffix}{temp_suffix}"
