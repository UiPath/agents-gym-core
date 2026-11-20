"""Agent initialisation: data preparation before a run.

An agent that needs local data before it can run (eval sets built from a dataset, a
retrieval index, ...) declares an ``init:`` block in ``agent.yaml``:

.. code-block:: yaml

    init:
      script: core::init.py
      data_dirs:                     # directories the script writes
        - evaluations
      params:                        # optional, passed to init()
        num_datapoints: 40
        seed: 0

``script`` is resolved like prompt and eval-set paths and must define
``init(agent_dir: Path, params: dict) -> None``. After a successful call, a
``.gym_init_done`` marker records a timestamp, the git commit and a hash of ``params``, so a
change to ``params`` (which can be overridden per version) triggers a new init. ``data_dirs``
is added to ``params``.

Running::

    gym init <agent_name>              # skips when already done
    gym init <agent_name> --force
    gym eval --agents <agent_name> --reinit
    UIPATH_GYM_SKIP_INIT=true gym eval --agents <agent_name>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agents_gym_core.project import ProjectDirectoryProvider

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from agents_gym_core.utils.directory_provider import DirectoryProvider

# DictConfig or ListConfig; only dict-style keys are read.
_Cfg = Any


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class AgentNotInitialisedError(RuntimeError):
    """Raised before execution when an agent requires data preparation."""

    def __init__(self, agent_name: str, agent_dir: Path, command: str = "gym") -> None:
        msg = textwrap.dedent(f"""
            Prepare local data for agent '{agent_name}':

                {agent_dir}

            Initialize the agent:

                {command} init {agent_name}

            Regenerate existing data:

                {command} init {agent_name} --force

            Or initialize before evaluation:

                {command} eval --agents {agent_name} --reinit

        """).strip()
        super().__init__(msg)
        self.agent_name = agent_name
        self.agent_dir = agent_dir


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


INIT_MARKER = ".gym_init_done"
"""Marker file written to the agent directory after a successful ``init()``."""


class AgentInit:
    """Parse and execute the ``init:`` block declared in an agent's ``agent.yaml``."""

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def has_init(cfg: DictConfig) -> bool:
        """Return ``True`` if the agent config contains an ``init:`` block."""
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, "init", default=None) is not None

    @staticmethod
    def uses_gepa_artifact(cfg: DictConfig) -> bool:
        """Return ``True`` if the agent's loop runs a program from gepa-artifact."""
        from omegaconf import OmegaConf

        loop = str(OmegaConf.select(cfg, "loop", default="") or "")
        return loop.startswith("dspy-program:gepa_artifact.")

    @staticmethod
    def get_data_dirs(cfg: _Cfg) -> list[str]:
        """Return ``init.data_dirs`` (or ``init.params.data_dirs``), or ``[]``."""
        from omegaconf import OmegaConf

        init_cfg = OmegaConf.select(cfg, "init", default=None)
        if init_cfg is None:
            return []

        data_dirs = OmegaConf.select(init_cfg, "data_dirs", default=None)
        if data_dirs is not None:
            return list(data_dirs)

        data_dirs = OmegaConf.select(init_cfg, "params.data_dirs", default=None)
        if data_dirs is not None:
            return list(data_dirs)

        return []

    @staticmethod
    def is_initialised(
        cfg: DictConfig,
        agent_name: str,
        agent_dir: Path,
        directory_provider: DirectoryProvider,
    ) -> bool:
        """Return ``True`` if there is no ``init:`` block or its marker matches ``params``."""
        if not AgentInit.has_init(cfg):
            return True

        _, params = AgentInit._resolve(cfg, agent_name, directory_provider)
        return AgentInit._check_marker(agent_dir, params)

    @staticmethod
    def check_or_raise(
        cfg: DictConfig,
        agent_name: str,
        agent_dir: Path,
        directory_provider: DirectoryProvider,
    ) -> None:
        """Raise :exc:`AgentNotInitialisedError` unless the agent is initialised.

        Called before eval-set files are opened, so a missing init gives an actionable error
        instead of a ``FileNotFoundError``. ``UIPATH_GYM_SKIP_INIT=true`` disables the check.
        """
        from agents_gym_core.utils.gym_settings_core import gym_settings_core

        if gym_settings_core.UIPATH_GYM_SKIP_INIT:
            return

        if not AgentInit.has_init(cfg):
            return

        if not AgentInit.is_initialised(cfg, agent_name, agent_dir, directory_provider):
            command = "gym"
            if isinstance(directory_provider, ProjectDirectoryProvider):
                command = f"gym --project {shlex.quote(str(directory_provider.get_project_root()))}"
            raise AgentNotInitialisedError(agent_name, agent_dir, command)

    @staticmethod
    def run(
        cfg: DictConfig,
        agent_name: str,
        agent_dir: Path,
        directory_provider: DirectoryProvider,
        force: bool = False,
    ) -> None:
        """Execute the agent's init script.

        Args:
            cfg: Full agent ``DictConfig`` (top-level, not just ``cfg.agent``).
            agent_name: Agent name — used for path resolution and logging.
            agent_dir: Resolved root directory of the agent.
            directory_provider: Resolves ``core::`` paths.
            force: When ``True``, run ``init()`` even if ``is_initialised``
                   returns ``True``.  Use for ``--force`` / ``--reinit``.
        """
        from agents_gym_core.utils.gym_settings_core import gym_settings_core

        if gym_settings_core.UIPATH_GYM_SKIP_INIT:
            print("  [gym init] UIPATH_GYM_SKIP_INIT=true — skipping")
            return

        if not AgentInit.has_init(cfg):
            print("  [gym init] no init: block in agent.yaml — nothing to do")
            return

        # Checked even when initialised: a wheel install has no submodule, and the cache that
        # replaces it can be deleted.
        if AgentInit.uses_gepa_artifact(cfg):
            from agents_gym_core.gepa_artifact import fetch_gepa_artifact

            print(f"  [gym init] gepa-artifact: {fetch_gepa_artifact()}")

        if not force:
            if AgentInit.is_initialised(cfg, agent_name, agent_dir, directory_provider):
                print("  [gym init] already initialised — skipping (use --force to re-run)")
                return

        # Remove stale marker before running so an interrupted init
        # doesn't leave us in a "looks initialised" state.
        AgentInit._remove_marker(agent_dir)

        script_path, params = AgentInit._resolve(cfg, agent_name, directory_provider)
        module = AgentInit._load_module(script_path)

        if not hasattr(module, "init"):
            raise AttributeError(
                f"Init script {script_path} must define a top-level "
                f"'init(agent_dir: Path, params: dict) -> None' function."
            )

        print(f"  [gym init] {script_path.name} params={params} ...")
        module.init(agent_dir, params)
        AgentInit._write_marker(agent_dir, params)
        print("  [gym init] done")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _params_hash(params: dict[str, Any]) -> str:
        """Deterministic hash of the params dict for marker comparison."""
        serialised = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    @staticmethod
    def _git_commit() -> str:
        """Return the current short git commit hash, or ``'unknown'``."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _write_marker(agent_dir: Path, params: dict[str, Any]) -> None:
        """Write the ``.gym_init_done`` marker after a successful ``init()``."""
        marker = agent_dir / INIT_MARKER
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": AgentInit._git_commit(),
            "params_hash": AgentInit._params_hash(params),
            "params": params,
        }
        marker.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    @staticmethod
    def _remove_marker(agent_dir: Path) -> None:
        """Delete the marker file if it exists."""
        marker = agent_dir / INIT_MARKER
        marker.unlink(missing_ok=True)

    @staticmethod
    def _check_marker(agent_dir: Path, params: dict[str, Any]) -> bool:
        """Return ``True`` if the marker exists and its ``params_hash`` matches *params*."""
        marker = agent_dir / INIT_MARKER
        if not marker.exists():
            return False
        try:
            data = json.loads(marker.read_text())
            return data.get("params_hash") == AgentInit._params_hash(params)
        except (json.JSONDecodeError, OSError):
            return False

    @staticmethod
    def _resolve(
        cfg: DictConfig,
        agent_name: str,
        directory_provider: DirectoryProvider,
    ) -> tuple[Path, dict[str, Any]]:
        """Return the resolved init script path and its params, with ``data_dirs`` added."""
        from omegaconf import OmegaConf

        init_cfg = OmegaConf.select(cfg, "init")
        if init_cfg is None:
            raise ValueError(f"Agent '{agent_name}' has no init: block in agent.yaml")

        script_placeholder: str | None = OmegaConf.select(init_cfg, "script", default=None)
        if not script_placeholder:
            raise ValueError(
                f"Agent '{agent_name}': init.script must be set (e.g. 'core::init.py')"
            )

        script_path = directory_provider.solve_path(agent_name, script_placeholder)
        if not script_path.exists():
            raise FileNotFoundError(f"Agent '{agent_name}': init script not found at {script_path}")

        raw_params = OmegaConf.select(init_cfg, "params", default=None)
        params: dict[str, Any] = (
            dict(OmegaConf.to_container(raw_params, resolve=True))  # type: ignore[arg-type]
            if raw_params is not None
            else {}
        )

        data_dirs = AgentInit.get_data_dirs(cfg)
        if data_dirs and "data_dirs" not in params:
            params["data_dirs"] = data_dirs

        return script_path, params

    @staticmethod
    def run_for_agent(
        agent_name: str,
        agent_version: str,
        registry: Any,
        force: bool = False,
    ) -> None:
        """Resolve the agent's config and directory from *registry*, then call :meth:`run`.

        Args:
            agent_name: Agent name.
            agent_version: Agent version string (e.g. ``"base"``).
            registry: A :class:`~agents_gym_core.hydra_agent.CoreAgentRegistry` instance.
            force: Passed through to :meth:`run`.
        """
        # Duck-typed registry: importing CoreAgentRegistry here would be circular.
        from omegaconf import OmegaConf

        # A registry can have several providers; use the one that owns the agent.
        providers = registry._get_providers()  # type: ignore[union-attr]
        directory_provider = next((p for p in providers if p._agent_exists(agent_name)), None)
        if directory_provider is None:
            searched = " or ".join(type(p).__name__ for p in providers)
            raise FileNotFoundError(f"Agent '{agent_name}' not found in {searched}")

        agent_dir = directory_provider.get_agent_dir(agent_name)
        cfg = registry._load_agent_base_config(agent_name, directory_provider)  # type: ignore[union-attr]

        # Merge version overrides so init.params can be overridden per version
        if agent_version not in (None, "", "base"):
            version_cfg = registry._load_agent_config_version(  # type: ignore[union-attr]
                agent_name, agent_version, directory_provider, raise_on_missing=False
            )
            if version_cfg is None:
                raise FileNotFoundError(
                    f"Version '{agent_version}' not found for agent '{agent_name}'"
                )
            from omegaconf import DictConfig

            # Struct mode off, so a version can add keys absent from agent.yaml.
            OmegaConf.set_struct(cfg, False)
            cfg = cast(DictConfig, OmegaConf.merge(cfg, version_cfg))

        AgentInit.run(cfg, agent_name, agent_dir, directory_provider, force=force)

    @staticmethod
    def _load_module(script_path: Path) -> Any:
        """Import *script_path* as an anonymous module via ``importlib``."""
        spec = importlib.util.spec_from_file_location("_gym_agent_init", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
