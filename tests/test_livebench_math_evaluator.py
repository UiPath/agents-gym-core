"""The LiveBench-Math evaluator scores answers off the event loop, one at a time."""

import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory

import agents_gym_core

AGENT_DIR = Path(agents_gym_core.__file__).parent / "agents" / "livebench_math"
SCORE_SECONDS = 0.25


def _load(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator(tmp_path: Path) -> Any:
    """The evaluator exactly as ``init.py`` writes it, built the way the gym loads it."""
    pytest.importorskip("gepa_artifact")
    init = _load(AGENT_DIR / "init.py", "livebench_math_init")
    custom = tmp_path / "evaluators" / "custom"
    custom.mkdir(parents=True)
    (custom / "livebench_math_evaluator.py").write_text(init.EVALUATOR_PY)
    return EvaluatorFactory.create_evaluator(init.EVALUATOR_JSON, evaluators_dir=custom.parent)


def test_scoring_keeps_the_event_loop_free_and_runs_one_answer_at_a_time(
    evaluator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator_module = sys.modules[type(evaluator).__module__]
    active = peak = 0
    guard = threading.Lock()

    def slow_score(question_d: dict, answer: str, debug: bool = False) -> tuple[int, str]:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(SCORE_SECONDS)
        with guard:
            active -= 1
        return 1, "ok"

    monkeypatch.setattr(evaluator_module, "calculate_livebench_score", slow_score)
    criteria = evaluator_module.LiveBenchMathCriteria(expectedOutput={"question_d": {"task": "t"}})
    output = types.SimpleNamespace(workload_output={"answer": "1"})

    async def run() -> tuple[list, float]:
        gaps: list[float] = []
        done = asyncio.Event()

        async def heartbeat() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.05)
        results = await asyncio.gather(*(evaluator.evaluate(output, criteria) for _ in range(4)))
        done.set()
        await beat
        return results, max(gaps)

    results, longest_pause = asyncio.run(run())
    assert [r.score for r in results] == [1.0] * 4
    assert peak == 1, "the scorer is not thread-safe; calls must not overlap"
    assert longest_pause < 2 * SCORE_SECONDS, "scoring blocked the event loop"
