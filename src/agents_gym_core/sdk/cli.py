"""The ``gym`` command.

``gym init``, ``gym eval`` (the testbed) and ``gym optimize`` cover the paper's workflow;
``gym reproduce`` evaluates the paper's prompts table by table.
``gym uipath generate|run|eval`` packages agents for the UiPath CLI (``uipath run`` /
``uipath eval``).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from agents_gym_core import _SHELL_ENV_KEYS
from agents_gym_core.paths import AGENTS_DIR, project_root

if TYPE_CHECKING:
    from agents_gym_core.utils.directory_provider import DirectoryProvider

# Imported by their subcommands: loading the agent stack takes seconds.
_PASSTHROUGH = {
    "eval": ("agents_gym_core.testbed_core", "evaluate agents on their eval sets (the testbed)"),
    "optimize": (
        "agents_gym_core.prompt_optimization.main",
        "optimize an agent's prompts with ReAPO",
    ),
    "reproduce": ("agents_gym_core.reproduce", "evaluate the paper's prompts, one table at a time"),
}
# Files `uipath init` writes for coding assistants; a developer's own copies are kept.
_UIPATH_INIT_DOCS = ("AGENTS.md", "CLAUDE.md")


def _load_config(cwd: Path) -> dict | None:
    """A ``gym_generate.json`` in the working directory points at other agents and a registry."""
    config_path = cwd / "gym_generate.json"
    return json.loads(config_path.read_text()) if config_path.exists() else None


def _project_context() -> tuple[Path, list[Path], str]:
    """(project root, agents directories, registry import line)."""
    cwd = Path.cwd()
    config = _load_config(cwd)
    if config:
        agents_dirs = [cwd / d for d in config.get("agents_dirs", [])]
        registry_import = config["registry_import"]
        root = cwd
    else:
        root = project_root()
        agents_dirs = [AGENTS_DIR]
        registry_import = "from agents_gym_core.hydra_agent import CoreAgentRegistry"
    return root, [d for d in agents_dirs if d.exists()], registry_import


def _find_agent_file(agents_dirs: list[Path], agent_name: str, *subpath: str) -> Path | None:
    for agents_dir in agents_dirs:
        path = agents_dir / agent_name / Path(*subpath)
        if path.exists():
            return path
    return None


def _uipath(*args: str) -> list[str]:
    exe = shutil.which("uipath", path=str(Path(sys.executable).parent))
    if exe is None:
        sys.exit("gym: the `uipath` CLI is not installed in this environment")
    # The uipath CLI sends usage telemetry unless this is set; gym leaves it off by default.
    os.environ.setdefault("UIPATH_TELEMETRY_ENABLED", "false")
    return [exe, *args]


def _build_registry(directory_provider: DirectoryProvider | None = None):
    if directory_provider is not None:
        return importlib.import_module("agents_gym_core.sdk.project_registry").ProjectAgentRegistry(
            directory_provider=directory_provider
        )
    cwd = Path.cwd()
    config = _load_config(cwd)
    if config and config.get("registry_import"):
        m = re.match(r"from\s+(\S+)\s+import\s+(\S+)", config["registry_import"])
        if not m:
            raise ValueError(f"Cannot parse registry_import: {config['registry_import']!r}")
        if str(cwd) not in sys.path:
            sys.path.insert(0, str(cwd))
        return getattr(importlib.import_module(m.group(1)), m.group(2))()

    from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry

    return CoreAgentRegistry()


# -----------------------------------------------------------------------------
# gym init
# -----------------------------------------------------------------------------


def _init(args: argparse.Namespace) -> None:
    from agents_gym_core.hydra_agent.agent_init import AgentInit

    directory_provider = args.directory_provider
    agents_dirs = (
        [directory_provider.get_agents_dir()] if directory_provider else _project_context()[1]
    )
    if args.list:
        _list_init_agents(agents_dirs, directory_provider)
        return
    if not args.agent_name:
        sys.exit("gym init: give an agent name, or --list")
    if not _find_agent_file(agents_dirs, args.agent_name):
        sys.exit(
            f"gym init: agent '{args.agent_name}' not found in {[str(d) for d in agents_dirs]}"
        )
    print(f"Initialising '{args.agent_name}:{args.version}' ...")
    try:
        AgentInit.run_for_agent(
            args.agent_name, args.version, _build_registry(directory_provider), force=args.force
        )
    except ModuleNotFoundError as exc:
        if (
            exc.name not in {"datasets", "dspy", "nltk"}
            or importlib.util.find_spec("dspy") is not None
        ):
            raise
        sys.exit(f"gym init: {exc}; the public benchmarks need `uv sync --locked --extra dspy`")


def _init_status(agent_yaml: Path, provider: DirectoryProvider | None = None) -> bool | None:
    """Whether an agent's init: data is in place; None when it declares no init: block."""
    from agents_gym_core.hydra_agent.agent_init import AgentInit

    config = importlib.import_module("omegaconf")
    raw_cfg = config.OmegaConf.load(agent_yaml)
    if config.OmegaConf.select(raw_cfg, "init", default=None) is None:
        return None
    cfg = (
        raw_cfg
        if isinstance(raw_cfg, config.DictConfig)
        else config.OmegaConf.create(dict(raw_cfg))
    )
    directory_provider = provider or _build_registry().get_directory_provider()
    return AgentInit.is_initialised(
        cfg, agent_yaml.parent.name, agent_yaml.parent, directory_provider
    )


def _list_init_agents(agents_dirs: list[Path], provider: DirectoryProvider | None = None) -> None:
    found_any = False
    for agents_dir in agents_dirs:
        for agent_yaml in sorted(agents_dir.glob("*/agent.yaml")):
            try:
                done = _init_status(agent_yaml, provider)
                status = "initialised" if done else "not initialised"
            except Exception as exc:
                done, status = False, f"error ({exc})"
            if done is None:
                continue
            found_any = True
            print(f"  {agent_yaml.parent.name:<44} {status}")
    if not found_any:
        print("  (no agents declare an init: block)")


def _uninitialised(agents: list[str], agents_dirs: list[Path]) -> list[str]:
    """Return agents whose declared data initialization is pending."""
    missing = []
    for agent in agents:
        agent_yaml = _find_agent_file(agents_dirs, agent, "agent.yaml")
        if agent_yaml and _init_status(agent_yaml) is False:
            missing.append(agent)
    return missing


# -----------------------------------------------------------------------------
# gym uipath run / eval
# -----------------------------------------------------------------------------


def _ensure_entrypoints(agent: str) -> None:
    """Ensure entry points include the agent and reflect its current configuration."""
    root, agents_dirs, registry_import = _project_context()
    langgraph, entry_points = root / "langgraph.json", root / "entry-points.json"
    graphs = json.loads(langgraph.read_text()).get("graphs", {}) if langgraph.exists() else {}
    configs = [_find_agent_file(agents_dirs, agent, f) for f in ("agent.yaml", "versions.yaml")]
    stale = (
        agent not in graphs
        or not entry_points.exists()
        or any(c and c.stat().st_mtime > entry_points.stat().st_mtime for c in configs)
    )
    if stale:
        if _uninitialised([agent], agents_dirs):
            sys.exit(f"gym uipath: run `gym init {agent}` first")
        buildable = {
            a
            for a in graphs
            if _find_agent_file(agents_dirs, a, "agent.yaml")
            and not _uninitialised([a], agents_dirs)
        }
        agents = sorted(buildable | {agent})
        print(f"Writing UiPath entry points for {agents}")
        _write_entrypoints(agents, root, registry_import)
        _uipath_init(root)


def _eval_set(agent: str, name: str | None, agents_dirs: list[Path], command: str) -> Path:
    """The named eval set, else the first one the agent's agent.yaml lists."""
    agent_yaml = _find_agent_file(agents_dirs, agent, "agent.yaml")
    if agent_yaml is None:
        sys.exit(f"{command}: agent '{agent}' not found")
    if _uninitialised([agent], agents_dirs):
        sys.exit(f"{command}: run `gym init {agent}` first")
    if name:
        path = agent_yaml.parent / "evaluations" / "eval-sets" / f"evaluation-set-{name}.json"
    else:
        refs = (yaml.safe_load(agent_yaml.read_text()).get("agent") or {}).get("evaluation_sets")
        if not refs:
            hint = "pass a JSON input" if command == "gym uipath run" else "pass --eval-set"
            sys.exit(f"{command}: '{agent}' lists no eval sets; {hint}")
        path = agent_yaml.parent / refs[0].split("::", 1)[-1]
    if not path.exists():
        sys.exit(f"{command}: no {path.name} for '{agent}'")
    return path


def _run(args: argparse.Namespace) -> None:
    root, agents_dirs, _ = _project_context()
    if args.input is not None and not args.input.isdigit():
        json_input = args.input
    else:
        index = int(args.input or 0)
        eval_set = _eval_set(args.agent_name, None, agents_dirs, "gym uipath run")
        evaluations = json.loads(eval_set.read_text()).get("evaluations", [])
        if index >= len(evaluations):
            sys.exit(f"gym uipath run: index {index} out of range ({len(evaluations)} datapoints)")
        print(f"Running datapoint {index} of {eval_set.name}: {evaluations[index].get('name')}")
        json_input = json.dumps(evaluations[index].get("inputs", {}))
    _ensure_entrypoints(args.agent_name)
    sys.exit(subprocess.run(_uipath("run", args.agent_name, json_input), cwd=root).returncode)


def _eval(args: argparse.Namespace) -> None:
    root, agents_dirs, _ = _project_context()
    eval_set = _eval_set(args.agent_name, args.eval_set, agents_dirs, "gym uipath eval")
    _ensure_entrypoints(args.agent_name)
    (root / "__uipath" / "state.db").unlink(missing_ok=True)
    results_dir = root / "experiments" / "sdk_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = _uipath("eval", args.agent_name, str(eval_set))
    if not args.report:
        cmd.append("--no-report")
    results = results_dir / f"eval-results-{eval_set.stem.removeprefix('evaluation-set-')}.json"
    cmd += ["--output-file", str(results), "--workers", str(args.workers)]
    sys.exit(subprocess.run(cmd, cwd=root).returncode)


# -----------------------------------------------------------------------------
# gym uipath generate
# -----------------------------------------------------------------------------


def _entrypoint(agent: str) -> str:
    """The entry point's function name: a Python identifier also for names such as ``my-agent``."""
    return re.sub(r"\W", "_", agent) + "_agent"


def _write_entrypoints(agents: list[str], root: Path, registry_import: str) -> None:
    (root / "langgraph.json").write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {a: f"./sdk_entrypoints.py:{_entrypoint(a)}" for a in agents},
                "env": ".env",
            },
            indent=2,
        )
        + "\n"
    )
    registry_class = registry_import.split()[-1]
    lines = [
        '"""SDK entry points, written by `gym uipath`."""',
        "",
        "from langgraph.graph import StateGraph",
        "",
        registry_import,
    ]
    for agent in agents:
        lines += [
            "",
            "",
            f"async def {_entrypoint(agent)}() -> StateGraph:",
            f'    return await {registry_class}().create_uncompiled_graph("{agent}")',
        ]
    (root / "sdk_entrypoints.py").write_text("\n".join(lines) + "\n")


def _copy_evaluations(agents: list[str], agents_dirs: list[Path], root: Path) -> None:
    eval_root = root / "evaluations"
    for sub in ("eval-sets", "evaluators"):
        (eval_root / sub).mkdir(parents=True, exist_ok=True)
        # Drop exact copies of any agent's files, left by an earlier run; the project's own stay.
        for f in (f for d in agents_dirs for f in d.glob(f"*/evaluations/{sub}/*.json")):
            for name in (f.name, f"{f.parents[2].name}-{f.name}"):
                copy = eval_root / sub / name
                if copy.is_file() and copy.read_bytes() == f.read_bytes():
                    copy.unlink()
    files: dict[str, list[tuple[str, Path]]] = {"eval-sets": [], "evaluators": []}
    for agent in agents:
        agent_evals = _find_agent_file(agents_dirs, agent, "evaluations")
        if not agent_evals:
            print(f"  Warning: no evaluations for agent '{agent}'")
            continue
        for sub in files:
            files[sub] += [(agent, f) for f in (agent_evals / sub).glob("*.json")]
    copied: dict[str, set[Path]] = {}
    for sub, sources in files.items():
        names = Counter(f.name for _, f in sources)
        copied[sub] = set()
        for agent, f in sources:
            # A file name several agents use gets the agent's name, so no copy overwrites another.
            copy = eval_root / sub / (f"{agent}-{f.name}" if names[f.name] > 1 else f.name)
            shutil.copy2(f, copy)
            copied[sub].add(copy)
    print(
        f"Copied {len(copied['eval-sets'])} eval sets and {len(copied['evaluators'])} evaluators "
        f"to {eval_root}"
    )


def _uipath_init(root: Path) -> None:
    for name in ("bindings.json", "entry-points.json", "uipath.json"):
        (root / name).unlink(missing_ok=True)
    kept = {
        name: (root / name).read_bytes() for name in _UIPATH_INIT_DOCS if (root / name).exists()
    }
    kept_diagrams = {
        path.name: path.read_bytes() for path in root.glob("*.mermaid") if path.is_file()
    }
    try:
        subprocess.run(_uipath("init"), cwd=root, check=True)
    finally:
        for name in _UIPATH_INIT_DOCS:
            if name in kept:
                (root / name).write_bytes(kept[name])
            else:
                (root / name).unlink(missing_ok=True)
        for mermaid in root.glob("*.mermaid"):
            if mermaid.name not in kept_diagrams and (mermaid.is_file() or mermaid.is_symlink()):
                mermaid.unlink()
        for name, content in kept_diagrams.items():
            (root / name).write_bytes(content)


def _generate(args: argparse.Namespace) -> None:
    root, agents_dirs, registry_import = _project_context()
    if not agents_dirs:
        sys.exit("gym uipath generate: no agents directory found")
    discovered = {p.parent.name for d in agents_dirs for p in d.glob("*/agent.yaml")}
    if args.agents:
        requested = {a.strip() for a in args.agents.split(",")}
        if unknown := requested - discovered:
            print(f"Warning: unknown agents ignored: {sorted(unknown)}", file=sys.stderr)
        agents = sorted(requested & discovered)
        if not agents:
            sys.exit("gym uipath generate: none of the requested agents exist")
        if missing := _uninitialised(agents, agents_dirs):
            sys.exit(f"gym uipath generate: run `gym init` first for {', '.join(missing)}")
    else:
        agents = sorted(discovered)
        if missing := _uninitialised(agents, agents_dirs):
            print(f"Skipping agents without their `gym init` data: {', '.join(missing)}")
            agents = [a for a in agents if a not in missing]
    print(f"Agents: {agents}")

    _write_entrypoints(agents, root, registry_import)
    print(f"Wrote {root / 'langgraph.json'} and {root / 'sdk_entrypoints.py'}")
    if args.with_evaluations:
        _copy_evaluations(agents, agents_dirs, root)
    if not args.no_init:
        _uipath_init(root)


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gym", description="Build, run, evaluate and optimize agents."
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="explicit standalone project directory (init, eval, optimize)",
    )
    parser.set_defaults(project_allowed=False, forwarded=False)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("new", help="create a standalone project for your own Python agent")
    p.add_argument("name", help="agent name")
    p.add_argument("--directory", help="new project directory (default: agent name)")
    p.set_defaults(func=_new)

    p = sub.add_parser("init", help="prepare an agent's data (splits, downloads, indexes)")
    p.add_argument("agent_name", nargs="?")
    p.add_argument("--version", default="base", help="version whose init params to use")
    p.add_argument("--force", action="store_true", help="re-run even if already initialised")
    p.add_argument("--list", action="store_true", help="list agents with an init: block")
    p.set_defaults(func=_init, project_allowed=True)

    for name, (module, help_text) in _PASSTHROUGH.items():
        p = sub.add_parser(name, help=f"{help_text} (gym {name} --help)", add_help=False)
        p.set_defaults(
            func=_forward, module=module, forwarded=True, project_allowed=name != "reproduce"
        )

    uipath = sub.add_parser("uipath", help="package and run agents with the UiPath CLI")
    usub = uipath.add_subparsers(dest="uipath_command", required=True, metavar="command")

    p = usub.add_parser("generate", help="write langgraph.json and SDK entry points, `uipath init`")
    p.add_argument("--agents", help="comma-separated agents (default: all)")
    p.add_argument(
        "--with-evaluations",
        action="store_true",
        help="copy eval sets and evaluators to ./evaluations",
    )
    p.add_argument("--no-init", action="store_true", help="skip `uipath init`")
    p.set_defaults(func=_generate)

    p = usub.add_parser("run", help="run an agent on one datapoint with `uipath run`")
    p.add_argument("agent_name")
    p.add_argument("input", nargs="?", help="datapoint index (default 0) or a JSON input")
    p.set_defaults(func=_run)

    p = usub.add_parser("eval", help="evaluate an agent on an eval set with `uipath eval`")
    p.add_argument("agent_name")
    p.add_argument("--eval-set", help="e.g. hover_test (default: the first in agent.yaml)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--report", action="store_true")
    p.set_defaults(func=_eval)
    return parser


def _new(args: argparse.Namespace) -> None:
    importlib.import_module("agents_gym_core.sdk.onboarding").new_project(args)


def _forward(args: argparse.Namespace) -> int | None:
    module = importlib.import_module(args.module)
    kwargs = {"directory_provider": args.directory_provider} if args.directory_provider else {}
    return module.main(argv=args.arguments, prog=f"gym {args.command}", **kwargs)


def main(argv: list[str] | None = None) -> int | None:
    parser = _parser()
    args, remaining = parser.parse_known_args(argv)
    if remaining and not args.forwarded:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if args.project and not args.project_allowed:
        use = "creates a project" if args.command == "new" else "uses bundled agents"
        parser.error(f"gym {args.command} {use}; omit --project")
    try:
        args.directory_provider = None
        if args.project:
            root = args.project.expanduser().resolve()
            if not os.environ.get("AGENTS_GYM_SKIP_DOTENV"):
                values = importlib.import_module("dotenv").dotenv_values(root / ".env")
                os.environ.update(
                    {k: v for k, v in values.items() if k not in _SHELL_ENV_KEYS and v is not None}
                )
            args.directory_provider = importlib.import_module(
                "agents_gym_core.project"
            ).ProjectDirectoryProvider(root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.arguments = remaining
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
