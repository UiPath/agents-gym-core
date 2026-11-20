"""The offline dependency gate runs before any installation or build hook."""

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependency_policy", ROOT / "scripts/check_dependency_policy.py"
)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)

FLOORS = {
    "pillow": "13.0.0",
    "pypdf": "6.21.4",
    "pyasn1": "0.6.5",
    "nltk": "3.10.4",
    "tornado": "6.6",
    "setuptools": "84.1.0",
    "black": "26.10.1",
    "nbconvert": "7.17.2",
    "opentelemetry-api": "1.45.1",
    "opentelemetry-sdk": "1.45.1",
    "langgraph-checkpoint-sqlite": "3.1.2",
    "uipath-langchain": "0.18.51",
    "mcp": "2.3.0",
    "mcp-types": "2.3.0",
    "pyjwt": "2.15.0",
}


@pytest.mark.parametrize(("name", "minimum"), FLOORS.items())
def test_installed_security_floor(name: str, minimum: str) -> None:
    assert Version(importlib.metadata.version(name)) >= Version(minimum)


def test_current_lock_matches_reviewed_policy() -> None:
    assert POLICY.check_policy(ROOT) == []


def test_ci_checks_the_minimal_optional_dependency_boundary() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        item
        for item in workflow["jobs"]["checks"]["steps"]
        if item.get("name") == "Core loop imports and types without DSPy"
    )
    command = step["run"]
    assert "--sync --extra dev" in command
    assert "--extra dspy" not in command
    assert "find_spec('dspy') is None" in command
    assert "import agents_gym_core.loops.loop_factory" in command
    assert "uv run --no-sync pyright src/agents_gym_core/loops" in command


def test_sync_disables_builds_except_reviewed_legacy_packages() -> None:
    command = POLICY.guarded_sync_command(ROOT, ["dspy", "dev"])
    guarded = {
        command[index + 1] for index, value in enumerate(command) if value == "--no-build-package"
    }
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    audit = json.loads((ROOT / POLICY.AUDIT).read_text())
    allowed = {"agents-gym-core"}
    for approval in audit["retained_source_only_packages"]:
        candidates = [item for item in lock["package"] if item["name"] == approval["name"]]
        if candidates and all(
            item["version"] == approval["version"]
            and item["source"].get("registry") == "https://pypi.org/simple"
            and not item.get("wheels")
            and item.get("sdist", {}).get("hash") == "sha256:" + approval["sha256"]
            for item in candidates
        ):
            allowed.add(approval["name"])
    assert guarded == {item["name"] for item in lock["package"]} - allowed
    assert command[:7] == ["uv", "sync", "--locked", "--extra", "dspy", "--extra", "dev"]


@pytest.fixture
def policy_fixture(tmp_path: Path) -> tuple[Path, dict]:
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    packages = [
        {
            "name": "pillow",
            "version": "13.0.0",
            "source": {"registry": "https://pypi.org/simple"},
            "wheels": [
                {
                    "url": "https://files.pythonhosted.org/pillow-13.0.0.whl",
                    "hash": "sha256:" + "a" * 64,
                }
            ],
        }
    ]
    approvals = [
        {"name": name, "version": "1.0", "sha256": "b" * 64}
        for name in ("antlr4-python3-runtime", "langdetect")
    ]
    packages.extend(
        {
            "name": item["name"],
            "version": item["version"],
            "source": {"registry": "https://pypi.org/simple"},
            "sdist": {
                "url": f"https://files.pythonhosted.org/{item['name']}-1.0.tar.gz",
                "hash": "sha256:" + item["sha256"],
            },
        }
        for item in approvals
    )
    audit = {
        "packages": [deepcopy(packages[0])],
        "changes": [{"package": "pillow", "after": "13.0.0"}],
        "reviewed_distributions": {"pillow": ["13.0.0"]},
        "candidate_lock_sha256": "",
        "retained_source_only_packages": approvals,
        "unchanged_external_sources": [],
    }
    (tmp_path / POLICY.AUDIT).parent.mkdir(parents=True)
    (tmp_path / POLICY.AUDIT).write_text(json.dumps(audit))
    lock = {"package": packages}
    _write_lock(tmp_path, lock)
    return tmp_path, lock


def _write_lock(root: Path, lock: dict) -> None:
    # Build the small subset read by the preflight; no TOML writer dependency is needed.
    lines = []
    for package in lock["package"]:
        lines += ["[[package]]", "name = " + json.dumps(package["name"])]
        if "version" in package:
            lines.append("version = " + json.dumps(package["version"]))
        source = ", ".join(
            key + " = " + json.dumps(value) for key, value in package["source"].items()
        )
        lines.append("source = {" + source + "}")
        for key in ("sdist", "wheels"):
            if key not in package:
                continue
            items = package[key] if key == "wheels" else [package[key]]
            rendered = [
                "{" + ", ".join(k + " = " + json.dumps(v) for k, v in item.items()) + "}"
                for item in items
            ]
            lines.append(
                key + " = " + ("[" + ",".join(rendered) + "]" if key == "wheels" else rendered[0])
            )
    (root / "uv.lock").write_text("\n".join(lines))
    audit = json.loads((root / POLICY.AUDIT).read_text())
    audit["candidate_lock_sha256"] = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    (root / POLICY.AUDIT).write_text(json.dumps(audit))


@pytest.mark.parametrize(
    "mutation", ["hash", "version", "missing", "origin", "new-source", "legacy-source"]
)
def test_preflight_refuses_unreviewed_changes(
    policy_fixture: tuple[Path, dict], mutation: str
) -> None:
    root, original = policy_fixture
    lock = deepcopy(original)
    package = next(item for item in lock["package"] if item["name"] == "pillow")
    if mutation == "hash":
        package["wheels"][0]["hash"] = "sha256:" + "0" * 64
    elif mutation == "version":
        package["version"] = "13.0.1"
    elif mutation == "missing":
        lock["package"].remove(package)
    elif mutation == "origin":
        package["wheels"][0]["url"] = "https://unexpected.invalid/package.whl"
    elif mutation == "new-source":
        lock["package"].append(
            {
                "name": "unreviewed-source",
                "version": "1.0",
                "source": {"registry": "https://pypi.org/simple"},
                "sdist": {
                    "url": "https://files.pythonhosted.org/unreviewed.tar.gz",
                    "hash": "sha256:" + "0" * 64,
                },
            }
        )
    else:
        retained = next(item for item in lock["package"] if item["name"] == "langdetect")
        retained["sdist"]["hash"] = "sha256:" + "0" * 64
    _write_lock(root, lock)
    assert POLICY.check_policy(root, check_snapshot=True)


def test_fixture_writer_preserves_policy_control(policy_fixture: tuple[Path, dict]) -> None:
    root, lock = policy_fixture
    _write_lock(root, lock)
    assert POLICY.check_policy(root) == []
    assert POLICY.check_policy(root, check_snapshot=True) == []


@pytest.mark.parametrize(
    "mutation",
    ["empty", "missing-pillow", "empty-both", "duplicate", "change-version", "hash-drift"],
)
def test_incomplete_snapshot_never_authorizes_sync(
    policy_fixture: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, lock = policy_fixture
    _write_lock(root, lock)
    audit = json.loads((root / POLICY.AUDIT).read_text())
    if mutation in {"empty", "empty-both"}:
        audit["packages"] = []
        if mutation == "empty-both":
            audit["changes"] = []
    elif mutation in {"missing-pillow", "hash-drift"}:
        audit["packages"] = [p for p in audit["packages"] if p["name"] != "pillow"]
        if mutation == "hash-drift":
            audit["candidate_lock_sha256"] = "0" * 64
    elif mutation == "duplicate":
        audit["packages"].append(audit["packages"][0])
    else:
        audit["changes"][0]["after"] = "0.0.0"
    (root / POLICY.AUDIT).write_text(json.dumps(audit))
    check = POLICY.check_policy
    monkeypatch.setattr(POLICY, "check_policy", lambda **kwargs: check(root, **kwargs))
    monkeypatch.setattr(sys, "argv", ["dependency-policy", "--check-snapshot", "--sync"])
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(POLICY.subprocess, "run", run)
    assert POLICY.main() == 1
    run.assert_not_called()


def test_lock_digest_must_match_reviewed_snapshot(policy_fixture: tuple[Path, dict]) -> None:
    root, lock = policy_fixture
    _write_lock(root, lock)
    with (root / "uv.lock").open("a") as stream:
        stream.write("\n# changed after review\n")
    assert POLICY.check_policy(root) == []
    assert any(
        "Lock snapshot drift" in error for error in POLICY.check_policy(root, check_snapshot=True)
    )


@pytest.mark.parametrize("name", sorted(POLICY.FORBIDDEN_DISTRIBUTIONS))
def test_removed_license_dependencies_stay_out(
    policy_fixture: tuple[Path, dict], name: str
) -> None:
    root, lock = policy_fixture
    package = deepcopy(next(p for p in lock["package"] if p["name"] == "pillow"))
    package["name"] = name
    lock["package"].append(package)
    _write_lock(root, lock)
    assert any("must not be reintroduced" in error for error in POLICY.check_policy(root))


@pytest.mark.parametrize(
    "script_name", ["check_dependency_policy.py", "refresh_dependency_audit.py"]
)
def test_unsupported_python_fails_before_tomllib_import(script_name: str) -> None:
    script = (
        "import runpy, sys; sys.version_info=(3, 10, 0); "
        f"runpy.run_path({str(ROOT / 'scripts' / script_name)!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Python 3.11" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("version", ["uv 0.11.26", "not uv", "uv 0.13.4-dev"])
def test_old_or_unidentified_uv_does_not_install(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(POLICY, "check_policy", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["dependency-policy", "--sync"])
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=version))
    monkeypatch.setattr(POLICY.subprocess, "run", run)
    assert POLICY.main() == 2
    run.assert_called_once_with(["uv", "--version"], capture_output=True, text=True)


def test_supported_uv_can_perform_guarded_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(POLICY, "check_policy", lambda **_: [])
    monkeypatch.setattr(sys, "argv", ["dependency-policy", "--sync", "--extra", "dev"])
    run = Mock(
        side_effect=[
            SimpleNamespace(returncode=0, stdout="uv 0.13.4 (reviewed)"),
            SimpleNamespace(returncode=0),
        ]
    )
    monkeypatch.setattr(POLICY.subprocess, "run", run)
    assert POLICY.main() == 0
    assert run.call_args_list[1].args[0] == POLICY.guarded_sync_command(ROOT, ["dev"])


def test_maintained_deepagents_api_is_pinned_exactly() -> None:
    from packaging.requirements import Requirement

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirement = next(
        Requirement(value)
        for value in project["project"]["dependencies"]
        if Requirement(value).name == "deepagents"
    )
    assert str(requirement.specifier) == "==0.7.26"


@pytest.mark.parametrize("change", ["version", "addition", "removal"])
def test_routine_public_lock_changes_do_not_require_snapshot_refresh(
    policy_fixture: tuple[Path, dict], change: str
) -> None:
    root, lock = policy_fixture
    wheel_package = next(item for item in lock["package"] if item["name"] == "pillow")
    if change == "version":
        wheel_package["version"] = "13.0.1"
    elif change == "addition":
        added = deepcopy(wheel_package)
        added["name"] = "public-helper"
        lock["package"].append(added)
    else:
        lock["package"].remove(wheel_package)
    _write_lock(root, lock)
    with (root / "uv.lock").open("a") as stream:
        stream.write("\n# native lock changed since the historical review\n")
    assert POLICY.check_policy(root) == []
    assert POLICY.check_policy(root, check_snapshot=True)


@pytest.mark.parametrize(
    "change",
    [
        "missing-hash",
        "invalid-hash",
        "origin",
        "no-artifacts",
        "external-source",
        "new-source-only",
        "legacy-hash",
        "legacy-version",
    ],
)
def test_light_policy_retains_structural_restrictions(
    policy_fixture: tuple[Path, dict], change: str
) -> None:
    root, lock = policy_fixture
    package = next(item for item in lock["package"] if item["name"] == "pillow")
    if change == "missing-hash":
        package["wheels"][0].pop("hash")
    elif change == "invalid-hash":
        package["wheels"][0]["hash"] = "sha256:not-a-digest"
    elif change == "origin":
        package["wheels"][0]["url"] = "https://unexpected.invalid/package.whl"
    elif change == "external-source":
        package["source"] = {"registry": "https://unexpected.invalid/simple"}
    elif change == "no-artifacts":
        package.pop("wheels")
        package.pop("sdist", None)
    elif change == "new-source-only":
        artifact = package.pop("wheels")[0]
        artifact["url"] = artifact["url"].replace(".whl", ".tar.gz")
        package["sdist"] = artifact
    else:
        legacy = next(item for item in lock["package"] if item["name"] == "langdetect")
        if change == "legacy-hash":
            legacy["sdist"]["hash"] = "sha256:" + "0" * 64
        else:
            legacy["version"] = "1.1.0"
    _write_lock(root, lock)
    assert POLICY.check_policy(root)


def test_legacy_name_does_not_allow_fallback_build_of_a_new_wheel_release(
    policy_fixture: tuple[Path, dict],
) -> None:
    root, lock = policy_fixture
    package = next(item for item in lock["package"] if item["name"] == "langdetect")
    package["version"] = "1.1.0"
    package["wheels"] = deepcopy(
        next(item for item in lock["package"] if item["name"] == "pillow")["wheels"]
    )
    _write_lock(root, lock)
    assert POLICY.check_policy(root) == []
    command = POLICY.guarded_sync_command(root, ["dev"])
    guarded = {
        command[index + 1] for index, item in enumerate(command) if item == "--no-build-package"
    }
    assert "langdetect" in guarded


def test_removing_an_approved_source_build_needs_no_snapshot_refresh(
    policy_fixture: tuple[Path, dict],
) -> None:
    root, lock = policy_fixture
    lock["package"] = [item for item in lock["package"] if item["name"] != "langdetect"]
    _write_lock(root, lock)
    assert POLICY.check_policy(root) == []
    assert POLICY.check_policy(root, check_snapshot=True)


def test_legacy_fork_cannot_grant_a_name_wide_source_build_exception(
    policy_fixture: tuple[Path, dict],
) -> None:
    root, lock = policy_fixture
    added = deepcopy(next(item for item in lock["package"] if item["name"] == "langdetect"))
    added["version"] = "1.1.0"
    added["wheels"] = deepcopy(
        next(item for item in lock["package"] if item["name"] == "pillow")["wheels"]
    )
    lock["package"].append(added)
    _write_lock(root, lock)
    assert POLICY.check_policy(root) == []
    command = POLICY.guarded_sync_command(root, [])
    assert any(
        command[index : index + 2] == ["--no-build-package", "langdetect"]
        for index in range(len(command) - 1)
    )


@pytest.mark.parametrize("strict", [False, True])
def test_cli_snapshot_check_is_explicit_opt_in(
    policy_fixture: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    root, lock = policy_fixture
    _write_lock(root, lock)
    with (root / "uv.lock").open("a") as stream:
        stream.write("\n# unrelated native lock metadata change\n")
    check = POLICY.check_policy
    monkeypatch.setattr(POLICY, "check_policy", lambda **kwargs: check(root, **kwargs))
    monkeypatch.setattr(
        sys, "argv", ["dependency-policy"] + (["--check-snapshot"] if strict else [])
    )
    assert POLICY.main() == (1 if strict else 0)
