# Dependency Security Policy

Install from the supported source checkout with Python 3.11.5 through 3.13 and uv
0.13.4 or newer. CI pins the tested uv 0.13.4. A stock macOS Python may be too old;
the tooling reports this before importing `tomllib`.
Packages come from PyPI, except the spaCy model `en_core_web_sm` (a GitHub release) and,
with the `huggingface` extra, torch; the lockfile applies uv's 2-day `exclude-newer` cooldown.

```bash
python3 scripts/check_dependency_policy.py
uv sync --locked --extra dspy --extra dev
```

The first command is an offline, standard-library-only preflight. It executes
before dependency installation and checks PyPI artifact origins and SHA256 hash
format, removed distributions, hashed build constraints, and exact approvals for
external sources and legacy source-only builds. Routine public wheel dependency
changes use this preflight alongside locked sync and maintainer review. The preflight
validates committed metadata; locked sync checks downloaded artifacts against their hashes.
Maintainer review covers publisher provenance and source changes.

Maintainers can opt into the stricter historical snapshot check:

```bash
python3 scripts/check_dependency_policy.py --check-snapshot
```

This also verifies the full lock digest, complete/nonduplicate reviewed cohort,
exact reviewed identities/artifacts and the current source-build list against
`docs/audits/security-dependencies.json`. `--check-snapshot` can be combined with
`--sync`; it is not enabled in CI's default contribution workflow.

For CI's stricter install policy, replace direct sync with:

```bash
python3 scripts/check_dependency_policy.py --sync --extra dspy --extra dev
```

The guarded sync checks the uv version, runs `uv sync --locked`, and applies the reviewed
per-package source-build allowlist on the current host. The repository's own editable build
remains enabled. Direct `uv sync --locked` uses uv's standard build selection.
The two retained legacy source archives (ANTLR runtime and langdetect) have fixed versions and SHA256 identities
in the audit snapshot; their setup metadata was reviewed separately.
Each source-build approval binds an exact locked version and archive hash. Refresh the
approval when either changes.

Isolated builds use the pinned, hashed build constraints in `pyproject.toml`.
Hash enforcement uses uv's experimental build-dependency hash feature; validate newer uv
releases before adopting them. Reuse build and download caches from trusted user and CI
scopes, since their ownership is part of the installation policy.

## Updating Dependencies

1. Choose compatible releases that retain or raise the current framework, provider and
   native-runtime package versions.
2. Review official release metadata, relevant advisories, release age, yank
   status, required Python/platform tags, and artifact SHA256 values. Prefer
   mature wheels; document any necessary security-release exception.
3. Regenerate `uv.lock` natively, preserving the immutable source/index policy.
   Review coupled SDK, protocol, instrumentation and optional-extra changes as
   coherent groups, preserving upstream transitive constraints.
4. When updating the reviewed artifact snapshot, use the dev environment to
   refresh the generated evidence against a
   locally available baseline commit/ref containing the previous `uv.lock`:

   ```bash
   uv run --no-sync python scripts/refresh_dependency_audit.py --baseline <baseline-ref>
   python3 scripts/check_dependency_policy.py --check-snapshot
   ```

   Omitting `--baseline` reuses the audit's recorded baseline, which keeps its review
   history. Pass `--baseline` (the release tag, or the commit before your lock change) only
   when that commit is not in your clone. The script reads official PyPI
   JSON, checks identity, artifact hashes/sizes/origins, yanks, release age (at least
   seven days), and currently reported advisories. It derives the changed cohort,
   package/change records, removals and exact lock digest. It writes the metadata snapshot
   atomically after policy validation and never installs or executes a package. Review the
   generated diff and maintain review notes
   in `manual_notes`. Refresh carries forward those notes and the public description while
   regenerating the artifact-review fields.
   Source/index changes, downgrades and new source-only archives require a separate manual
   review. Exact AWS SDK
   and hf-xet companion pins in `pyproject.toml` move with the audit, not independently.
   Downgrades are checked per existing native resolution branch before reusing
   artifacts, so lower-Python branch upgrades do not compare against an unrelated
   higher branch. Repartitioned, unknown or ambiguous branch labels require a
   separate manual downgrade review.
5. Run the preflight, relevant contract tests and supported Python CI matrix.
   Run version-floor checks in the candidate environment. Use local provider/model stand-ins
   (`tests/fixtures/openai_stub_server.py`) for routine validation.

The snapshot records review-time evidence. Review current advisories, publisher provenance
and source changes alongside it. Resolve dependency alerts through compatible patched
releases that satisfy upstream constraints, then confirm default-branch ingestion and
rescanning.

## Runtime Boundaries

- Core's MCP server uses stdio for optimizer communication.
- Workbook validation is scoped to the vector-ingestion paths and resource budgets in
  [bounded workbooks](bounded-workbooks.md).
- Validate optional Hugging Face and native/GPU runtimes on supported hosts. Review model
  and checkpoint provenance alongside runtime compatibility.
