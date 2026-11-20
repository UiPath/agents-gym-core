"""MIPROv2 demos that quote AIME and LiveBench-Math are filled in by ``gym init``.

The repository keeps only references to the datapoints; ``write_demos`` must rebuild the
paper's demo files byte for byte from the generated eval sets.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agents_gym_core.utils.benchmark_demos import DATAPOINT, write_demos

AGENTS = Path(__file__).resolve().parents[1] / "src" / "agents_gym_core" / "agents"
REFS = "prompts/miprov2/predict.refs.json"
# SHA-256 of the MIPROv2 demo files.
PAPER_DEMOS_SHA256 = {
    "aime_2025_agent": "1daa720b228a7287b44f66fbb482d34576a35155912482c3e4b421adf3d69a2d",
    "livebench_math": "f94b3943bc6f4c80bb4e742e2180a6d5aef7d69386d30c388d1a62d1d412053b",
}
QUOTED_FIELDS = {"problem", "question", "solution"}


def test_references_are_replaced_by_the_datapoint_text(tmp_path: Path) -> None:
    eval_sets = tmp_path / "evaluations" / "eval-sets"
    eval_sets.mkdir(parents=True)
    item = {"name": "dp-1", "inputs": {"problem": "Find x."}, "criteria": {"solution": "x = 2"}}
    (eval_sets / "set.json").write_text(json.dumps({"evaluations": [item]}))
    demo = {
        "problem": {DATAPOINT: "dp-1", "$path": ["inputs", "problem"]},
        "reasoning": "model-written",
        "solution": {DATAPOINT: "dp-1", "$path": ["criteria", "solution"]},
        "answer": "2",
    }
    (tmp_path / "predict.refs.json").write_text(json.dumps({"demos": [demo]}))

    out = write_demos(tmp_path, "predict.refs.json")

    assert out == tmp_path / "predict.json"
    assert json.loads(out.read_text()) == {
        "demos": [
            {"problem": "Find x.", "reasoning": "model-written", "solution": "x = 2", "answer": "2"}
        ]
    }


@pytest.mark.parametrize("agent", sorted(PAPER_DEMOS_SHA256))
def test_tracked_demos_quote_no_benchmark_text(agent: str) -> None:
    demos = json.loads((AGENTS / agent / REFS).read_text())["demos"]
    assert demos
    for demo in demos:
        for field in QUOTED_FIELDS & demo.keys():
            assert DATAPOINT in demo[field], f"{agent}: {field} is quoted, not referenced"


@pytest.mark.parametrize("agent", sorted(PAPER_DEMOS_SHA256))
def test_init_rebuilds_the_papers_demos(agent: str, tmp_path: Path) -> None:
    eval_sets = AGENTS / agent / "evaluations" / "eval-sets"
    if not any(eval_sets.glob("*.json")):
        pytest.skip(f"{agent}: generated splits missing; run `uv run gym init {agent}`")
    shutil.copytree(eval_sets, tmp_path / "evaluations" / "eval-sets")
    (tmp_path / REFS).parent.mkdir(parents=True)
    shutil.copy(AGENTS / agent / REFS, tmp_path / REFS)

    out = write_demos(tmp_path, REFS)

    assert hashlib.sha256(out.read_bytes()).hexdigest() == PAPER_DEMOS_SHA256[agent]
