"""Few-shot demos whose benchmark text ``gym init`` fills in.

The MIPROv2 demos of AIME and LiveBench-Math quote competition problems and solutions, which
this repository does not redistribute. Their tracked ``*.refs.json`` file keeps everything else
(the model-written reasoning, the answers) and replaces each quoted text with the datapoint it
comes from::

    "problem": {"$datapoint": "aime-2022-2024-025", "$path": ["inputs", "problem"]}

After building the eval sets from the downloaded datasets, the agent's ``init.py`` calls
``write_demos``, which writes the demo file the agent's versions load, with each reference
replaced by that datapoint's text.
"""

import json
from pathlib import Path
from typing import Any

DATAPOINT = "$datapoint"
PATH = "$path"
REFS_SUFFIX = ".refs.json"


def _resolve(value: Any, datapoints: dict[str, dict]) -> Any:
    if isinstance(value, dict) and DATAPOINT in value:
        node: Any = datapoints[value[DATAPOINT]]
        for key in value[PATH]:
            node = node[key]
        return node
    if isinstance(value, dict):
        return {key: _resolve(item, datapoints) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, datapoints) for item in value]
    return value


def write_demos(agent_dir: Path, refs: str) -> Path:
    """Write ``refs`` (a ``*.refs.json`` path relative to ``agent_dir``) with its references
    filled in from the agent's eval sets, next to it without ``.refs``; return the written path."""
    refs_path = agent_dir / refs
    if not refs_path.name.endswith(REFS_SUFFIX):
        raise ValueError(f"{refs_path} is not a {REFS_SUFFIX} file")
    datapoints: dict[str, dict] = {}
    for eval_set in sorted((agent_dir / "evaluations" / "eval-sets").glob("*.json")):
        for item in json.loads(eval_set.read_text())["evaluations"]:
            datapoints.setdefault(item["name"], item)
    payload = _resolve(json.loads(refs_path.read_text()), datapoints)
    out = refs_path.with_name(refs_path.name.removesuffix(REFS_SUFFIX) + ".json")
    # The paper's demo files: two-space indent, ASCII escapes, no trailing newline.
    out.write_text(json.dumps(payload, indent=2))
    return out
