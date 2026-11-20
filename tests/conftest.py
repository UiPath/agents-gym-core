"""Keep tests hermetic: never use the developer's ``.env``.

Its provider keys would otherwise reach the code under test and send stubbed runs to a real
model (Azure OpenAI wins over the local stub's ``OPENAI_BASE_URL``). Three loaders read it:
our own ``load_env_file`` (skipped through ``AGENTS_GYM_SKIP_DOTENV``, which subprocesses
inherit), litellm, which loads it on import unless ``LITELLM_MODE`` is not ``DEV``, and the
UiPath SDK, whose ``uipath._cli`` loads ``<cwd>/.env`` with ``override=True`` when first
imported. Import it here once and undo what it loaded.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values

os.environ["AGENTS_GYM_SKIP_DOTENV"] = "1"
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

_before = dict(os.environ)
import uipath._cli  # noqa: E402, F401

for _key in dotenv_values(Path.cwd() / ".env"):
    if _key in _before:
        os.environ[_key] = _before[_key]
    else:
        os.environ.pop(_key, None)


@pytest.fixture(autouse=True)
def _no_optimizer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimizers export their workspace and target for the MCP server subprocess; undo
    that after each test so a later test is not treated as part of an optimization run."""
    for key in ("COACH_TARGET_AGENT", "COACH_OUTPUT_DIR", "COACH_MAX_CHARS"):
        monkeypatch.delenv(key, raising=False)


STUB = Path(__file__).resolve().parent / "fixtures" / "openai_stub_server.py"


@pytest.fixture
def stub(tmp_path: Path) -> Iterator[tuple[int, Path]]:
    """A local OpenAI-compatible stub: (port, log of the requests it answered)."""
    log = tmp_path / "requests.jsonl"
    proc = subprocess.Popen(
        [sys.executable, "-I", str(STUB)],
        stdout=subprocess.PIPE,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "STUB_LOG": str(log)},
    )
    assert proc.stdout is not None
    port = int(proc.stdout.readline())
    try:
        yield port, log
    finally:
        proc.terminate()
        proc.wait(timeout=10)
