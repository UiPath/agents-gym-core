"""Check reviewed lock artifacts before installation; optionally perform a guarded sync."""

import sys

if sys.version_info < (3, 11):  # noqa: UP036 - run before tomllib on stock macOS Python.
    sys.stderr.write(
        "Dependency tooling requires Python 3.11 or newer; use python3.11+ or uv's managed Python.\n"
    )
    raise SystemExit(2)

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("docs/audits/security-dependencies.json")
MIN_UV_VERSION = (0, 13, 4)
FORBIDDEN_DISTRIBUTIONS = frozenset({"aiostream", "frozendict", "func-timeout", "levenshtein"})


def _artifacts(package: dict) -> list[dict]:
    return package.get("wheels", []) + ([package["sdist"]] if package.get("sdist") else [])


def check_policy(
    root: Path = ROOT, *, audit: dict | None = None, check_snapshot: bool = False
) -> list[str]:
    """Return policy errors using committed metadata only, without executing packages."""
    lock_bytes = (root / "uv.lock").read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if audit is None:
        audit = dict(json.loads((root / AUDIT).read_text()))
    reviewed = {(p["name"], p["version"]): p for p in audit["packages"]} if check_snapshot else {}
    retained = {p["name"]: p for p in audit["retained_source_only_packages"]}
    external = audit["unchanged_external_sources"]
    errors: list[str] = []
    if check_snapshot:
        changed = {(p["package"], p["after"]) for p in audit["changes"]}
        declared = {
            (name, version)
            for name, versions in audit["reviewed_distributions"].items()
            for version in versions
        }
        if (
            set(reviewed) != declared
            or len(reviewed) != len(audit["packages"])
            or set(reviewed) != changed
            or len(changed) != len(audit["changes"])
        ):
            errors.append("Incomplete or inconsistent reviewed dependency cohort")
        if hashlib.sha256(lock_bytes).hexdigest() != audit["candidate_lock_sha256"]:
            errors.append(f"Lock snapshot drift; refresh {AUDIT}")
    if project["build-system"]["requires"] != ["setuptools==84.1.0"]:
        errors.append("Build backend must use reviewed setuptools==84.1.0")
    if not project["tool"]["uv"].get("require-build-hashes"):
        errors.append("Isolated build dependencies must require reviewed hashes")
    constraints = project["tool"]["uv"].get("build-constraint-dependencies", [])
    if not any(
        isinstance(c, dict) and c.get("requirement") == "setuptools==84.1.0" and c.get("hashes")
        for c in constraints
    ):
        errors.append("Missing hashed setuptools build constraint")
    seen_reviewed: set[tuple[str, str]] = set()
    for package in lock["package"]:
        name = package["name"]
        if name in FORBIDDEN_DISTRIBUTIONS:
            errors.append(f"Removed dependency must not be reintroduced: {name}")
        if not re.fullmatch(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", name):
            errors.append("Invalid locked distribution name")
            continue
        source = package["source"]
        if source.get("editable") == "." and name == "agents-gym-core":
            continue
        if source.get("registry") != "https://pypi.org/simple":
            candidate = {
                k: package[k]
                for k in ("name", "version", "source", "wheels", "sdist")
                if k in package
            }
            if candidate not in external:
                errors.append(f"Unreviewed external source: {name}")
            continue
        artifacts = _artifacts(package)
        if not artifacts:
            errors.append(f"Missing public distribution artifacts: {name}")
        for artifact in artifacts:
            parsed = urlsplit(artifact["url"])
            if parsed.scheme != "https" or parsed.netloc != "files.pythonhosted.org":
                errors.append(f"Unexpected public artifact origin: {name}")
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", artifact.get("hash", "")):
                errors.append(f"Missing artifact SHA256: {name}")
        if package.get("sdist") and not package.get("wheels"):
            allowed = retained.get(name)
            if (
                not allowed
                or package["version"] != allowed["version"]
                or package["sdist"]["hash"] != "sha256:" + allowed["sha256"]
            ):
                errors.append(f"Unreviewed source-only distribution: {name}")
        key = (name, package["version"])
        if key in reviewed:
            seen_reviewed.add(key)
            record = reviewed[key]
            if (
                record["source"] != source
                or record.get("wheels", []) != package.get("wheels", [])
                or record.get("sdist") != package.get("sdist")
            ):
                errors.append(f"Reviewed artifact drift: {name}; refresh {AUDIT}")
    for name, version in reviewed.keys() - seen_reviewed:
        errors.append(f"Reviewed distribution missing: {name}=={version}")
    source_only = {
        p["name"]
        for p in lock["package"]
        if p["source"].get("registry") == "https://pypi.org/simple"
        and p.get("sdist")
        and not p.get("wheels")
    }
    if check_snapshot and set(retained) != source_only:
        errors.append("Source-build review list is stale; refresh the audit")
    return errors


def guarded_sync_command(root: Path, extras: list[str], uv: str = "uv") -> list[str]:
    """Permit only the reviewed legacy source builds; all other packages need wheels."""
    audit = json.loads((root / AUDIT).read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    retained = {p["name"]: p for p in audit["retained_source_only_packages"]}
    allowed = {"agents-gym-core"}
    for name, approval in retained.items():
        candidates = [package for package in lock["package"] if package["name"] == name]
        if candidates and all(
            package["version"] == approval["version"]
            and package["source"].get("registry") == "https://pypi.org/simple"
            and not package.get("wheels")
            and package.get("sdist", {}).get("hash") == "sha256:" + approval["sha256"]
            for package in candidates
        ):
            allowed.add(name)
    command = [uv, "sync", "--locked"]
    for extra in extras:
        command.extend(["--extra", extra])
    for name in sorted({p["name"] for p in lock["package"]} - allowed):
        command.extend(["--no-build-package", name])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument(
        "--check-snapshot",
        action="store_true",
        help="Also verify the exact historical audit cohort and full lock digest",
    )
    args = parser.parse_args()
    try:
        errors = check_policy(check_snapshot=args.check_snapshot)
    except (OSError, KeyError, ValueError, TypeError):
        print("Dependency policy metadata is missing or malformed")
        return 1
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Dependency policy passed" + (" (exact snapshot verified)" if args.check_snapshot else ""),
        flush=True,
    )
    if args.sync:
        try:
            version = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        except OSError:
            print("uv 0.13.4 or newer is required for guarded sync; CI uses 0.13.4")
            return 2
        match = re.match(r"^uv (\d+)\.(\d+)\.(\d+)(?:\s|$)", version.stdout or "")
        if version.returncode or not match or tuple(map(int, match.groups())) < MIN_UV_VERSION:
            print("uv 0.13.4 or newer is required for hashed build dependencies; CI uses 0.13.4")
            return 2
        return subprocess.run(guarded_sync_command(ROOT, args.extra), cwd=ROOT).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
