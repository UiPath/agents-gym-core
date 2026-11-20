"""A source checkout keeps writing under the repo; an installed wheel writes under the cwd."""

from pathlib import Path

import pytest

from agents_gym_core import paths

REPO = Path(__file__).resolve().parents[1]


def test_a_source_checkout_uses_the_repo() -> None:
    assert paths.source_checkout() == REPO
    assert paths.project_root() == REPO
    assert paths.AGENTS_DIR == REPO / "src" / "agents_gym_core" / "agents"


def test_an_installed_package_uses_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "site-packages" / "agents_gym_core"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(paths, "PACKAGE_DIR", site_packages)
    monkeypatch.chdir(tmp_path)
    assert paths.source_checkout() is None
    assert paths.project_root() == tmp_path
