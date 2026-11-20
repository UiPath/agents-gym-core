"""Pluggable model-registry overrides, configured by environment variable.

Core resolves models through its own registries (``models/chat/clients.py``). A deployment
that needs different endpoints, such as a private Azure deployment or a vLLM server, registers
them without core importing it: set ``AGENTS_GYM_MODEL_OVERRIDES`` to one or more
comma-separated ``module:function`` entries. Each function is called once, with no
arguments, before models are resolved. Subprocesses (testbed runs launched by the optimizer,
the MCP server) inherit the variable, so every process applies the same overrides.

Example::

    AGENTS_GYM_MODEL_OVERRIDES=my_package.models:apply_model_overrides
"""

import importlib
import os

ENV_VAR = "AGENTS_GYM_MODEL_OVERRIDES"

_applied: set[str] = set()


def apply_configured_model_overrides() -> list[str]:
    """Call each hook named in ``AGENTS_GYM_MODEL_OVERRIDES`` once; return those applied now."""
    applied: list[str] = []
    for entry in (part.strip() for part in os.environ.get(ENV_VAR, "").split(",")):
        if not entry or entry in _applied:
            continue
        module_name, sep, func_name = entry.partition(":")
        if not sep or not module_name or not func_name:
            raise ValueError(f"{ENV_VAR} entry {entry!r} must look like 'package.module:function'")
        getattr(importlib.import_module(module_name), func_name)()
        _applied.add(entry)
        applied.append(entry)
    return applied
