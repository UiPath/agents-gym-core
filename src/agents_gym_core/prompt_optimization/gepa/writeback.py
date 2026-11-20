"""Persist a GEPA-evolved candidate as a new agent version.

Writes ``prompts/<version>/system_prompt.md`` (and the user prompt the optimizer ran with, when
given), a version folder like the agent's other versions (so an optimizer run cannot read it),
and adds an entry to ``versions.yaml`` (rewriting the file) so the new prompt is a runnable
version (alongside ``base``). The user can then evaluate it via
``gym eval --agents <name>:<version>``.

The directory provider resolves where the agent lives; the new version's prompt path uses
the ``core::`` prefix.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from agents_gym_core.utils.directory_provider import DirectoryProvider


def _resolve_agent_dir(directory_provider: DirectoryProvider, agent_name: str) -> Path:
    """Find the agent directory on disk, falling back to ``CoreDirectoryProvider`` when the
    given provider does not know the agent."""
    try:
        return directory_provider.get_agent_dir(agent_name)
    except FileNotFoundError:
        from agents_gym_core.utils.directory_provider import CoreDirectoryProvider

        return CoreDirectoryProvider().get_agent_dir(agent_name)


def save_evolved_version(
    *,
    directory_provider: DirectoryProvider,
    agent_name: str,
    version_name: str,
    evolved_system_prompt: str,
    source_version: str = "base",
    user_prompt: str | None = None,
) -> tuple[Path, Path]:
    """Write the evolved system_prompt, preserving the source execution overrides.

    ``user_prompt``, when given, is saved next to it and pinned in the new version, so the
    version keeps the user prompt the optimizer ran with even after the agent's files change.

    Evaluation sets come from the base agent, rather than the optimizer's selected split.

    Returns ``(prompt_path, versions_yaml_path)``.
    """
    agent_dir = _resolve_agent_dir(directory_provider, agent_name)
    versions_path = agent_dir / "versions.yaml"
    if versions_path.exists():
        loaded = OmegaConf.load(versions_path)
        if not isinstance(loaded, DictConfig):
            loaded = OmegaConf.create({})
        versions = loaded
    else:
        versions = OmegaConf.create({})

    source_entry = OmegaConf.create({})
    if source_version != "base":
        source = versions.get(source_version)
        if not isinstance(source, DictConfig):
            raise ValueError(f"Source version {source_version!r} not found in {versions_path}")
        source_entry = source.copy()
        source_entry.pop("defaults", None)
        source_agent = source_entry.get("agent")
        if isinstance(source_agent, DictConfig):
            source_agent.pop("evaluation_sets", None)
    agent_entry = {"system_prompt": f"core::prompts/{version_name}/system_prompt.md"}
    if user_prompt is not None:
        agent_entry["user_prompt"] = f"core::prompts/{version_name}/user_prompt.md"
    new_entry = OmegaConf.merge(source_entry, {"version": version_name, "agent": agent_entry})
    prompt_path = agent_dir / "prompts" / version_name / "system_prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(evolved_system_prompt, encoding="utf-8")
    if user_prompt is not None:
        (prompt_path.parent / "user_prompt.md").write_text(user_prompt, encoding="utf-8")
    versions[version_name] = new_entry
    OmegaConf.save(versions, versions_path)
    return prompt_path, versions_path
