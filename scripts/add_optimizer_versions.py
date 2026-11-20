"""Give every per-optimizer prompt set of the public benchmarks a ``<dir>-test`` version.

Each prompt set ``prompts/<dir>/`` (``reapo__gpt-4.1-mini``, ``reapo__opus-4.6``,
``reapo__qwen3.5-9b``, the ablation sets, ``gepa__gpt-5``, ...) gets a
version that copies the agent's ``reapo__gpt-5-test`` (same user prompt and held-out test
split) with its prompts taken from ``prompts/<dir>/`` (``<module>_predict.md`` when a GEPA
export names files that way). Existing versions are left alone, so the script is idempotent::

    python scripts/add_optimizer_versions.py
"""

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / "src" / "agents_gym_core" / "agents"
BENCHMARKS = ["hotpotqa", "hover", "aime_2025_agent", "livebench_math", "ifbench", "papillon"]
TEMPLATE = "reapo__gpt-5-test"
TEMPLATE_DIR = "reapo__gpt-5"
HEADER = "# Per-optimizer prompt sets: one <dir>-test version per prompts/<dir> (scripts/add_optimizer_versions.py)"


def _retarget(ref: str, agent_dir: Path, prompt_dir: str) -> str:
    prefix = f"core::prompts/{TEMPLATE_DIR}/"
    if not ref.startswith(prefix):
        return ref
    name = ref.removeprefix(prefix)
    for candidate in (name, name.removesuffix(".md") + "_predict.md"):
        if (agent_dir / "prompts" / prompt_dir / candidate).is_file():
            return f"core::prompts/{prompt_dir}/{candidate}"
    raise SystemExit(f"{agent_dir.name}: prompts/{prompt_dir} has no file for {name}")


def versions_for(agent: str) -> dict[str, dict]:
    agent_dir = AGENTS / agent
    existing = yaml.safe_load((agent_dir / "versions.yaml").read_text()) or {}
    template = existing[TEMPLATE]["agent"]
    added: dict[str, dict] = {}
    for prompt_dir in sorted(p.name for p in (agent_dir / "prompts").iterdir() if p.is_dir()):
        if not prompt_dir.startswith(("reapo__", "gepa__")):
            continue
        name = f"{prompt_dir}-test"
        if name in existing:
            continue
        spec = copy.deepcopy(template)
        for field in ("system_prompt", "user_prompt"):
            if field in spec:
                spec[field] = _retarget(spec[field], agent_dir, prompt_dir)
        spec["signature_prompts"] = {
            module: _retarget(ref, agent_dir, prompt_dir)
            for module, ref in (spec.get("signature_prompts") or {}).items()
        } or None
        if spec["signature_prompts"] is None:
            del spec["signature_prompts"]
        added[name] = {"version": name, "agent": spec}
    return added


def main() -> None:
    for agent in BENCHMARKS:
        added = versions_for(agent)
        if not added:
            print(f"{agent}: up to date")
            continue
        path = AGENTS / agent / "versions.yaml"
        text = path.read_text().rstrip("\n")
        if HEADER not in text:
            text += "\n\n" + HEADER
        block = yaml.safe_dump(added, sort_keys=False, default_flow_style=False)
        path.write_text(text + "\n" + block)
        print(f"{agent}: added {', '.join(added)}")


if __name__ == "__main__":
    main()
