"""Refresh generated lock/PyPI evidence, preserving manually maintained review notes."""

import sys

if sys.version_info < (3, 11):  # noqa: UP036 - actionable error before importing tomllib.
    sys.stderr.write(
        "Dependency tooling requires Python 3.11 or newer; use a supported development interpreter.\n"
    )
    raise SystemExit(2)

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("docs/audits/security-dependencies.json")
MIN_RELEASE_AGE_DAYS = 7


def _artifacts(package: dict) -> list[dict]:
    return package.get("wheels", []) + ([package["sdist"]] if package.get("sdist") else [])


def _identity(package: dict) -> tuple[str, str, str]:
    return package["name"], package["version"], json.dumps(package["source"], sort_keys=True)


def _artifact_identity(package: dict) -> list[tuple]:
    return sorted((item["url"], item["hash"], item.get("size")) for item in _artifacts(package))


def _resolution_branches(lock: dict) -> frozenset[str]:
    markers = lock.get("resolution-markers")
    if markers is None:
        markers = [
            marker
            for package in lock["package"]
            for marker in package.get("resolution-markers", [])
        ]
    if not isinstance(markers, list) or any(
        type(marker) is not str or not marker for marker in markers
    ):
        raise ValueError("Malformed resolution branches need separate downgrade review")
    return frozenset(markers) or frozenset({""})


def _branch_versions(lock: dict, branches: frozenset[str]) -> dict[tuple[str, str], str]:
    selections: dict[tuple[str, str], str] = {}
    for package in lock["package"]:
        markers = package.get("resolution-markers", list(branches))
        if not isinstance(markers, list) or not markers or not set(markers) <= branches:
            raise ValueError("Unknown resolution branches need separate downgrade review")
        for branch in markers:
            key = package["name"], branch
            if key in selections:
                raise ValueError(
                    f"Ambiguous resolution branch needs separate downgrade review: {key[0]}"
                )
            selections[key] = package["version"]
    return selections


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "dependency_audit_policy", Path(__file__).with_name("check_dependency_policy.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline(root: Path, reference: str) -> tuple[str, dict]:
    environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", reference + "^{commit}"],
        cwd=root,
        env=environment,
        text=True,
        stderr=subprocess.DEVNULL,  # a missing ref is reported by the caller
    ).strip()
    raw = subprocess.check_output(["git", "show", commit + ":uv.lock"], cwd=root, env=environment)
    return commit, tomllib.loads(raw.decode())


def _fetch_release(name: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
    with urlopen(url, timeout=30) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError(f"PyPI metadata response is unexpectedly large: {name}")
    return json.loads(raw)


def rebuild_audit(
    root: Path,
    baseline: dict,
    baseline_commit: str,
    *,
    fetch_release: Callable[[str, str], dict] = _fetch_release,
    now: datetime | None = None,
) -> dict:
    """Return complete generated evidence; never install or execute a dependency."""
    try:
        from packaging.utils import canonicalize_name
        from packaging.version import Version
    except ImportError:
        raise ValueError(
            "Use a synced dev interpreter: uv run --no-sync python scripts/refresh_dependency_audit.py"
        ) from None

    policy = _load_policy()
    current = tomllib.loads((root / "uv.lock").read_text())
    existing = json.loads((root / AUDIT).read_text())
    now = now or datetime.now(UTC)
    # Native fork labels are stable across ordinary updates. Refuse repartitioning
    # instead of approximating arbitrary marker intersections or ignoring a branch.
    branches = _resolution_branches(baseline)
    if branches != _resolution_branches(current):
        raise ValueError("Resolution branches changed; separate downgrade review is required")
    previous_selections = _branch_versions(baseline, branches)
    for key, version in _branch_versions(current, branches).items():
        if key in previous_selections and Version(version) < Version(previous_selections[key]):
            raise ValueError(f"Dependency downgrade needs separate review: {key[0]}")
    old = {_identity(package): package for package in baseline["package"]}
    new = {_identity(package): package for package in current["package"]}
    prior_by_name: dict[str, list[dict]] = {}
    for package in baseline["package"]:
        prior_by_name.setdefault(package["name"], []).append(package)
    prior_sources = {
        name: {json.dumps(package["source"], sort_keys=True) for package in packages}
        for name, packages in prior_by_name.items()
    }
    source_reviews = {p["name"]: p for p in existing["retained_source_only_packages"]}
    retained, external, packages, changes = [], [], [], []
    for key, package in sorted(new.items()):
        name, version, source_key = key
        source = package["source"]
        if name in policy.FORBIDDEN_DISTRIBUTIONS:
            raise ValueError(f"Removed dependency must not be reintroduced: {name}")
        if source.get("editable") == "." and name == "agents-gym-core":
            continue
        if name in prior_sources and source_key not in prior_sources[name]:
            raise ValueError(f"Dependency source changed; review it separately: {name}")
        if source.get("registry") != "https://pypi.org/simple":
            if key not in old or _artifact_identity(package) != _artifact_identity(old[key]):
                raise ValueError(f"External source/artifact changed; review it separately: {name}")
            external.append(
                {
                    k: package[k]
                    for k in ("name", "version", "source", "wheels", "sdist")
                    if k in package
                }
            )
            continue
        if package.get("sdist") and not package.get("wheels"):
            review = source_reviews.get(name)
            if (
                not review
                or review["version"] != version
                or "sha256:" + review["sha256"] != package["sdist"]["hash"]
            ):
                raise ValueError(f"Source-only distribution needs a manual archive review: {name}")
            retained.append({key: review[key] for key in ("name", "version", "sha256")})
        if key in old and _artifact_identity(package) == _artifact_identity(old[key]):
            continue
        previous = sorted({p["version"] for p in prior_by_name.get(name, [])}, key=Version)
        if not package.get("wheels"):
            raise ValueError(f"New source-only selection needs a manual review: {name}")
        release = fetch_release(name, version)
        if canonicalize_name(release["info"]["name"]) != name or Version(
            release["info"]["version"]
        ) != Version(version):
            raise ValueError(f"PyPI identity mismatch: {name}")
        active = [
            item.get("id")
            for item in (release.get("vulnerabilities") or [])
            if not item.get("withdrawn")
        ]
        if active:
            raise ValueError(f"Active PyPI advisories require review: {name}: {active}")
        official = {item["url"]: item for item in release["urls"]}
        youngest = float("inf")
        for artifact in _artifacts(package):
            item = official.get(artifact["url"])
            if (
                not item
                or item.get("yanked")
                or artifact["hash"] != "sha256:" + item["digests"]["sha256"]
                or artifact.get("size") != item["size"]
            ):
                raise ValueError(f"Artifact hash, origin, size or yank mismatch: {name}")
            uploaded = datetime.fromisoformat(item["upload_time_iso_8601"].replace("Z", "+00:00"))
            age = (now - uploaded).total_seconds() / 86400
            if age < MIN_RELEASE_AGE_DAYS:
                raise ValueError(f"Artifact is younger than {MIN_RELEASE_AGE_DAYS} days: {name}")
            youngest = min(youngest, age)
        packages.append(
            {
                "name": name,
                "version": version,
                "source": source,
                "sdist": package.get("sdist"),
                "wheels": package.get("wheels", []),
                "requires_python": release["info"].get("requires_python"),
                "youngest_artifact_days_at_review": round(youngest, 3),
                "active_pypi_advisories_at_review": [],
            }
        )
        changes.append(
            {
                "package": name,
                "before": previous,
                "after": version,
                "artifacts": len(_artifacts(package)),
                "youngest_days": round(youngest, 2),
            }
        )
    result = {
        key: copy.deepcopy(existing[key])
        for key in ("description", "manual_notes")
        if key in existing
    }
    result.update(
        schema_version=1,
        reviewed_at=now.isoformat(),
        baseline_commit=baseline_commit,
        candidate_lock_sha256=hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        packages=packages,
        changes=changes,
        reviewed_distributions={
            name: sorted({p["version"] for p in packages if p["name"] == name})
            for name in sorted({p["name"] for p in packages})
        },
        retained_source_only_packages=retained,
        unchanged_external_sources=external,
        removed_distributions=[
            {"name": key[0], "version": key[1]}
            for key in sorted(old.keys() - new.keys())
            if key[0] not in {p["name"] for p in current["package"]}
        ],
        sources_unchanged=True,
        dependency_downgrades=False,
    )
    errors = policy.check_policy(root, audit=result, check_snapshot=True)
    if errors:
        raise ValueError("\n".join(errors))
    return result


def write_audit(root: Path, audit: dict) -> None:
    target = root / AUDIT
    if target.is_symlink():
        raise ValueError("The audit destination must not be a symlink")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as stream:
            temporary = stream.name
            json.dump(audit, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", help="Locally available Git commit/ref containing the baseline uv.lock"
    )
    arguments = parser.parse_args()
    try:
        existing = json.loads((ROOT / AUDIT).read_text())
        reference = arguments.baseline or existing["baseline_commit"]
        try:
            commit, baseline = _baseline(ROOT, reference)
        except subprocess.CalledProcessError:
            if arguments.baseline:
                raise
            print(
                f"Audit refresh failed: the recorded baseline {reference} is not in this clone; "
                "pass --baseline <ref>, the commit whose uv.lock the audit was reviewed against.",
                file=sys.stderr,
            )
            return 1
        audit = rebuild_audit(ROOT, baseline, commit)
        write_audit(ROOT, audit)
    except (OSError, KeyError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Audit refresh failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Refreshed {AUDIT}: {len(audit['packages'])} reviewed distributions; manual notes retained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
