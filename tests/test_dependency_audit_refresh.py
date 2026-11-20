"""Regenerate audit evidence from synthetic releases without executing dependencies."""

import copy
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_refresh", ROOT / "scripts/refresh_dependency_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
REFRESH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFRESH)
NOW = datetime(2026, 11, 5, tzinfo=UTC)


def package(name: str, version: str, *, source: dict | None = None) -> dict:
    return {
        "name": name,
        "version": version,
        "source": source or {"registry": "https://pypi.org/simple"},
        "wheels": [
            {
                "url": f"https://files.pythonhosted.org/{name}-{version}.whl",
                "hash": "sha256:" + "a" * 64,
                "size": 10,
            }
        ],
    }


def write_lock(root: Path, packages: list[dict], branches: list[str] | None = None) -> None:
    lines = [] if branches is None else ["resolution-markers=" + json.dumps(branches)]
    for item in packages:
        lines += [
            "[[package]]",
            "name=" + json.dumps(item["name"]),
            "version=" + json.dumps(item["version"]),
            "source={" + ",".join(k + "=" + json.dumps(v) for k, v in item["source"].items()) + "}",
        ]
        if "resolution-markers" in item:
            lines.append("resolution-markers=" + json.dumps(item["resolution-markers"]))
        for kind in ("wheels", "sdist"):
            if kind not in item:
                continue
            artifacts = item[kind] if kind == "wheels" else [item[kind]]
            rendered = [
                "{" + ",".join(k + "=" + json.dumps(v) for k, v in a.items()) + "}"
                for a in artifacts
            ]
            lines.append(
                kind + "=" + ("[" + ",".join(rendered) + "]" if kind == "wheels" else rendered[0])
            )
    (root / "uv.lock").write_text("\n".join(lines))


def release(item: dict) -> dict:
    return {
        "info": {"name": item["name"], "version": item["version"], "requires_python": ">=3.11"},
        "urls": [
            {
                "url": a["url"],
                "digests": {"sha256": a["hash"].split(":")[1]},
                "size": a["size"],
                "yanked": False,
                "upload_time_iso_8601": "2026-10-01T00:00:00Z",
            }
            for a in item.get("wheels", [])
        ],
        "vulnerabilities": [],
    }


@pytest.fixture
def audit_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    target = tmp_path / REFRESH.AUDIT
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "baseline_commit": "a" * 40,
                "reviewed_at": "2026-10-02T00:00:00Z",
                "manual_notes": {"owner": "keep this manual decision"},
                "internal_metadata": {"ticket": "not public review evidence"},
                "retained_source_only_packages": [],
                "unchanged_external_sources": [],
                "packages": [],
                "changes": [],
                "reviewed_distributions": {},
            }
        )
    )
    return tmp_path


def test_refresh_derives_new_cohort_without_code_edits_and_keeps_notes(audit_root: Path) -> None:
    candidate = [package("demo", "2.0"), package("another-helper", "1.0")]
    write_lock(audit_root, candidate)
    lookup = {(p["name"], p["version"]): release(p) for p in candidate}
    result = REFRESH.rebuild_audit(
        audit_root,
        {"package": [package("demo", "1.0")]},
        "b" * 40,
        fetch_release=lambda n, v: lookup[n, v],
        now=NOW,
    )
    assert result["reviewed_distributions"] == {"another-helper": ["1.0"], "demo": ["2.0"]}
    assert len(result["packages"]) == len(result["changes"]) == 2
    assert result["manual_notes"] == {"owner": "keep this manual decision"}
    assert "internal_metadata" not in result
    assert (
        result["candidate_lock_sha256"]
        == hashlib.sha256((audit_root / "uv.lock").read_bytes()).hexdigest()
    )
    REFRESH.write_audit(audit_root, result)
    assert json.loads((audit_root / REFRESH.AUDIT).read_text()) == result


@pytest.mark.parametrize(
    "mutation", ["hash", "size", "yanked", "young", "advisory", "identity", "origin"]
)
def test_refresh_refuses_unreviewable_release(audit_root: Path, mutation: str) -> None:
    candidate = package("demo", "2.0")
    write_lock(audit_root, [candidate])
    metadata = release(candidate)
    if mutation == "hash":
        metadata["urls"][0]["digests"]["sha256"] = "0" * 64
    elif mutation == "size":
        metadata["urls"][0]["size"] = 20
    elif mutation == "yanked":
        metadata["urls"][0]["yanked"] = True
    elif mutation == "young":
        metadata["urls"][0]["upload_time_iso_8601"] = NOW.isoformat()
    elif mutation == "advisory":
        metadata["vulnerabilities"] = [{"id": "SYNTHETIC-ADVISORY", "withdrawn": None}]
    elif mutation == "identity":
        metadata["info"]["name"] = "someone-else"
    else:
        candidate["wheels"][0]["url"] = "https://unexpected.invalid/demo.whl"
        write_lock(audit_root, [candidate])
        metadata = release(candidate)
    original = (audit_root / REFRESH.AUDIT).read_bytes()
    with pytest.raises(ValueError):
        REFRESH.rebuild_audit(
            audit_root, {"package": []}, "b" * 40, fetch_release=lambda *_: metadata, now=NOW
        )
    assert (audit_root / REFRESH.AUDIT).read_bytes() == original


def test_unchanged_records_need_no_network_and_removals_are_recorded(audit_root: Path) -> None:
    retained = package("demo", "2.0")
    write_lock(audit_root, [retained])
    result = REFRESH.rebuild_audit(
        audit_root,
        {"package": [retained, package("old-helper", "1.0")]},
        "b" * 40,
        fetch_release=lambda *_: pytest.fail("unchanged record fetched"),
        now=NOW,
    )
    assert result["packages"] == result["changes"] == []
    assert result["reviewed_distributions"] == {}
    assert result["removed_distributions"] == [{"name": "old-helper", "version": "1.0"}]


def test_refresh_will_not_admit_new_source_only_code(audit_root: Path) -> None:
    source = package("source-helper", "1.0")
    source["sdist"] = source.pop("wheels")[0]
    write_lock(audit_root, [source])
    with pytest.raises(ValueError, match="manual archive review"):
        REFRESH.rebuild_audit(audit_root, {"package": []}, "b" * 40, now=NOW)


def test_source_approval_exports_only_public_identity_fields(audit_root: Path) -> None:
    source = package("legacy-helper", "1.0")
    source["sdist"] = source.pop("wheels")[0]
    write_lock(audit_root, [source])
    public = {"name": source["name"], "version": source["version"], "sha256": "a" * 64}
    target = audit_root / REFRESH.AUDIT
    existing = json.loads(target.read_text())
    existing["retained_source_only_packages"] = [
        {**public, "internal_notes": {"ticket": "do not export"}}
    ]
    target.write_text(json.dumps(existing))
    result = REFRESH.rebuild_audit(
        audit_root,
        {"package": [source]},
        "b" * 40,
        fetch_release=lambda *_: pytest.fail("unchanged source must not fetch"),
        now=NOW,
    )
    assert result["retained_source_only_packages"] == [public]


@pytest.mark.parametrize("kind", ["source", "external-artifact", "downgrade"])
def test_refresh_preserves_sources_and_versions(audit_root: Path, kind: str) -> None:
    before = package("demo", "2.0")
    candidate = copy.deepcopy(before)
    if kind == "source":
        candidate["source"] = {"registry": "https://unexpected.invalid/simple"}
    elif kind == "external-artifact":
        before["source"] = candidate["source"] = {"url": "https://example.invalid/fixed.whl"}
        candidate["wheels"][0]["hash"] = "sha256:" + "0" * 64
    else:
        candidate = package("demo", "1.0")
    write_lock(audit_root, [candidate])
    with pytest.raises(ValueError):
        REFRESH.rebuild_audit(audit_root, {"package": [before]}, "b" * 40, now=NOW)


BRANCHES = ["python_full_version < '3.12'", "python_full_version >= '3.12'"]


def forked_package(version: str, marker: str) -> dict:
    result = package("branch-demo", version)
    result["resolution-markers"] = [marker]
    return result


@pytest.mark.parametrize("change", ["lower-upgrade", "artifact-refresh", "marker-order"])
def test_refresh_compares_versions_within_their_resolution_branch(
    audit_root: Path, change: str
) -> None:
    low = forked_package("1.0", BRANCHES[0])
    high = forked_package("2.0", BRANCHES[1])
    candidate = copy.deepcopy(low)
    if change == "lower-upgrade":
        candidate = forked_package("1.1", BRANCHES[0])
    elif change == "artifact-refresh":
        wheel = copy.deepcopy(candidate["wheels"][0])
        wheel["url"] = wheel["url"].replace(".whl", "-platform.whl")
        candidate["wheels"].append(wheel)
    write_lock(audit_root, [candidate, high], list(reversed(BRANCHES)))
    result = REFRESH.rebuild_audit(
        audit_root,
        {"package": [low, high], "resolution-markers": BRANCHES},
        "b" * 40,
        fetch_release=lambda *_: release(candidate),
        now=NOW,
    )
    expected = {} if change == "marker-order" else {"branch-demo": [candidate["version"]]}
    assert result["reviewed_distributions"] == expected
    assert result["dependency_downgrades"] is False


@pytest.mark.parametrize(
    "change",
    ["universal-rollback", "swap", "lower-rollback", "repartition", "unknown", "ambiguous"],
)
def test_refresh_never_reuses_artifacts_before_branch_downgrade_review(
    audit_root: Path, change: str
) -> None:
    low = forked_package("1.0", BRANCHES[0])
    high = forked_package("2.0", BRANCHES[1])
    candidate = [copy.deepcopy(low), copy.deepcopy(high)]
    branches = BRANCHES
    if change == "universal-rollback":
        candidate = [package("branch-demo", "1.0")]
    elif change == "swap":
        candidate[0]["resolution-markers"], candidate[1]["resolution-markers"] = (
            [BRANCHES[1]],
            [BRANCHES[0]],
        )
    elif change == "lower-rollback":
        candidate[0] = forked_package("0.9", BRANCHES[0])
    elif change == "repartition":
        branches = ["python_full_version >= '3.11'"]
        candidate = [forked_package("1.0", branches[0])]
    elif change == "unknown":
        candidate[0]["resolution-markers"] = ["os_name == 'posix'"]
    else:
        candidate[0].pop("resolution-markers")
    write_lock(audit_root, candidate, branches)
    with pytest.raises(ValueError, match="downgrade"):
        REFRESH.rebuild_audit(
            audit_root,
            {"package": [low, high], "resolution-markers": BRANCHES},
            "b" * 40,
            fetch_release=lambda *_: pytest.fail("branch checks must precede artifact reuse/fetch"),
            now=NOW,
        )
