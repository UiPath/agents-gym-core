"""Agents Gym Core: run, evaluate and optimize LLM agents with ReAPO.

Code for *Simplicity is Enough: ReAct Agents for Prompt Optimization* (NeurIPS 2026).
"""

__version__ = "1.0.0"

import os as _os
import warnings as _warnings

# Raised inside uipath-langchain, which imports a module the UiPath SDK deprecated.
_warnings.filterwarnings("ignore", "uipath.platform.common.constants is deprecated", FutureWarning)

# litellm loads the nearest .env when imported unless LITELLM_MODE is set to something other
# than DEV, so with AGENTS_GYM_SKIP_DOTENV it would still bring the provider keys back.
if _os.environ.get("AGENTS_GYM_SKIP_DOTENV"):
    _os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

# The UiPath SDK's ``uipath._cli`` loads ``<cwd>/.env`` with ``override=True`` when first
# imported. Trigger that load here and restore the shell's values, so variables set in the
# shell take precedence over .env. With AGENTS_GYM_SKIP_DOTENV, also drop the variables that
# only the .env defined.
_shell_env = dict(_os.environ)
_SHELL_ENV_KEYS = frozenset(_shell_env)
import uipath._cli  # noqa: E402, F401

if _os.environ.get("AGENTS_GYM_SKIP_DOTENV"):
    for _key in set(_os.environ) - set(_shell_env):
        del _os.environ[_key]
_os.environ.update(_shell_env)
del _shell_env

from agents_gym_core.gepa_artifact import ensure_gepa_artifact_importable  # noqa: E402

ensure_gepa_artifact_importable()

__all__ = ["__version__"]
