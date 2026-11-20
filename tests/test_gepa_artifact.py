"""gepa-artifact is found as the submodule, or cloned at its commit by ``gym init``."""

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from agents_gym_core import gepa_artifact
from agents_gym_core.hydra_agent.agent_init import AgentInit

REPO = Path(__file__).resolve().parents[1]


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def test_the_pinned_commit_is_the_submodules() -> None:
    try:
        entry = _git(REPO, "ls-tree", "HEAD", "gepa-artifact")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert entry.split()[2] == gepa_artifact.GEPA_ARTIFACT_COMMIT


@pytest.fixture
def no_submodule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No GEPA_ARTIFACT_DIR, no submodule checkout, and the cache under ``tmp_path``."""
    monkeypatch.delenv("GEPA_ARTIFACT_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(gepa_artifact, "_submodule_dir", lambda: tmp_path / "missing")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return tmp_path


def test_root_prefers_the_override_then_the_submodule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submodule = tmp_path / "gepa-artifact"
    (submodule / "gepa_artifact").mkdir(parents=True)
    monkeypatch.setattr(gepa_artifact, "_submodule_dir", lambda: submodule)
    monkeypatch.setenv("GEPA_ARTIFACT_DIR", str(tmp_path / "elsewhere"))
    assert gepa_artifact.gepa_artifact_root() == (tmp_path / "elsewhere").resolve()
    monkeypatch.delenv("GEPA_ARTIFACT_DIR")
    assert gepa_artifact.gepa_artifact_root() == submodule


def test_without_a_submodule_the_root_is_the_cache(no_submodule: Path) -> None:
    root = gepa_artifact.gepa_artifact_root()
    assert root.parent == no_submodule / "cache" / "agents-gym-core"


def test_fetch_clones_the_pinned_commit(
    no_submodule: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = no_submodule / "upstream"
    (upstream / "gepa_artifact").mkdir(parents=True)
    (upstream / "gepa_artifact" / "__init__.py").write_text("")
    _git(upstream, "init", "-q")
    _git(upstream, "config", "uploadpack.allowReachableSHA1InWant", "true")
    _git(upstream, "add", ".")
    _git(upstream, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c")
    monkeypatch.setattr(gepa_artifact, "GEPA_ARTIFACT_REPO", str(upstream))
    monkeypatch.setattr(gepa_artifact, "GEPA_ARTIFACT_COMMIT", _git(upstream, "rev-parse", "HEAD"))

    root = gepa_artifact.fetch_gepa_artifact()

    assert root == gepa_artifact.gepa_artifact_root()
    assert (root / "gepa_artifact" / "__init__.py").is_file()
    assert _git(root, "rev-parse", "HEAD") == gepa_artifact.GEPA_ARTIFACT_COMMIT
    assert str(root) in sys.path
    assert gepa_artifact.fetch_gepa_artifact() == root  # found, not cloned again


def test_a_missing_override_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEPA_ARTIFACT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="GEPA_ARTIFACT_DIR"):
        gepa_artifact.fetch_gepa_artifact()


@pytest.mark.parametrize(
    ("loop", "expected"),
    [
        ("dspy-program:gepa_artifact.benchmarks.hover.hover_program.HoverMultiHop", True),
        ("basic", False),
        (None, False),
    ],
)
def test_init_fetches_only_for_gepa_programs(loop: str | None, expected: bool) -> None:
    cfg = OmegaConf.create({"loop": loop} if loop else {})
    assert AgentInit.uses_gepa_artifact(cfg) is expected


def test_every_tracked_resource_type_is_package_data() -> None:
    """A wheel holds only the file types listed in pyproject's package-data."""
    import tomllib

    try:
        files = _git(REPO, "ls-files", "src/agents_gym_core").splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    setuptools = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]
    packaged = {p.rsplit(".", 1)[-1] for p in setuptools["package-data"]["agents_gym_core"]}
    resources = {Path(f).suffix.lstrip(".") for f in files} - {"py"}
    assert resources <= packaged
