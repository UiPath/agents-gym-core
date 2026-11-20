"""Locate the gepa-artifact checkout that holds the public benchmarks' DSPy programs.

The public-benchmark agents (hotpotqa, hover, aime_2025_agent, livebench_math, ifbench,
papillon) run GEPA's released DSPy programs (``loop: dspy-program:gepa_artifact...``).
The programs (and IFBench's data) are versioned in the repo-root git submodule; Core's
``dspy`` extra supplies their runtime dependencies through the lockfile. For an installed
wheel, ``gym init`` clones the same commit into ``~/.cache/agents-gym-core``. Set
``GEPA_ARTIFACT_DIR`` to use a checkout elsewhere.
"""

import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

GEPA_ARTIFACT_REPO = "https://github.com/gepa-ai/gepa-artifact.git"
# The submodule's commit; tests/test_gepa_artifact.py keeps the two equal.
GEPA_ARTIFACT_COMMIT = "cbefbc1aa0f43dd39874ec4bf42211365dbda42e"


# The Wikipedia-abstracts archive GEPA's HoVer program builds its BM25 index from, at the
# dspy/cache revision that added it.
WIKI_ABSTRACTS_URL = (
    "https://huggingface.co/dspy/cache/resolve/ef6a5e72a98b47cef31574a400fea8fe149559a3/"
    "wiki.abstracts.2017.tar.gz"
)
WIKI_ABSTRACTS_SHA256 = "744183e61af986bde9b25c880b59c1502618a8b673671e189cbc0ee684fceb42"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_wiki_abstracts(url: str) -> None:
    """Replacement for ``dspy.utils.download`` while GEPA builds its index: fetch the pinned
    archive into the working directory and check its SHA-256."""
    filename = os.path.basename(WIKI_ABSTRACTS_URL)
    if os.path.basename(url) != filename:
        raise ValueError(f"unexpected download {url!r}")
    if os.path.exists(filename) and _sha256(filename) == WIKI_ABSTRACTS_SHA256:
        return
    print(f"Downloading {WIKI_ABSTRACTS_URL} ...")
    with urllib.request.urlopen(WIKI_ABSTRACTS_URL) as response, open(filename, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    if _sha256(filename) != WIKI_ABSTRACTS_SHA256:
        os.remove(filename)
        raise ValueError(f"{filename}: SHA-256 differs from {WIKI_ABSTRACTS_SHA256}")


@contextlib.contextmanager
def verified_wiki_abstracts() -> Iterator[None]:
    """While active, GEPA's corpus download uses the pinned, checked archive, and tar
    extraction keeps every member inside its target directory (``tarfile.data_filter``)."""
    import dspy.utils

    # The raw class attribute, so a staticmethod default is restored as a staticmethod.
    download = dspy.utils.download
    extraction_filter = inspect.getattr_static(tarfile.TarFile, "extraction_filter")
    dspy.utils.download = _download_wiki_abstracts
    tarfile.TarFile.extraction_filter = staticmethod(tarfile.data_filter)
    try:
        yield
    finally:
        dspy.utils.download = download
        tarfile.TarFile.extraction_filter = extraction_filter


def _submodule_dir() -> Path:
    # src/agents_gym_core/gepa_artifact.py -> repo root
    return Path(__file__).resolve().parents[2] / "gepa-artifact"


def _cache_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "agents-gym-core" / f"gepa-artifact-{GEPA_ARTIFACT_COMMIT[:12]}"


def _is_checkout(root: Path) -> bool:
    return (root / "gepa_artifact").is_dir()


def gepa_artifact_root() -> Path:
    """Return the gepa-artifact checkout: ``$GEPA_ARTIFACT_DIR``, else ``<repo>/gepa-artifact``
    when the submodule is checked out, else the cache directory ``fetch_gepa_artifact`` fills."""
    override = os.environ.get("GEPA_ARTIFACT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    submodule = _submodule_dir()
    return submodule if _is_checkout(submodule) else _cache_dir()


def fetch_gepa_artifact() -> Path:
    """Return the gepa-artifact checkout, cloning it at ``GEPA_ARTIFACT_COMMIT`` into the cache
    when none is found. Git checks every fetched object against the pinned commit."""
    root = gepa_artifact_root()
    if not _is_checkout(root):
        if os.environ.get("GEPA_ARTIFACT_DIR"):
            raise FileNotFoundError(f"GEPA_ARTIFACT_DIR={root} holds no gepa_artifact package")
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".gepa-artifact-", dir=root.parent))
        try:
            _clone_at_commit(tmp)
            try:
                tmp.rename(root)
            except OSError:
                if not _is_checkout(root):  # else another process cloned it first
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # a no-op after the rename
    ensure_gepa_artifact_importable()
    return root


def _clone_at_commit(target: Path) -> None:
    # Omit the repo's 1.9 GB archived experiment results.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=target, env=env, check=True, capture_output=True, text=True
        )
        return done.stdout.strip()

    git("init", "-q")
    git("remote", "add", "origin", GEPA_ARTIFACT_REPO)
    git("fetch", "-q", "--depth", "1", "origin", GEPA_ARTIFACT_COMMIT)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
    if (head := git("rev-parse", "HEAD")) != GEPA_ARTIFACT_COMMIT:
        raise RuntimeError(f"gepa-artifact checked out {head}, not {GEPA_ARTIFACT_COMMIT}")


def skip_spacy_model_download() -> None:
    """Use the locked ``en_core_web_sm`` dependency for IFBench's spaCy model.

    The ``dspy`` extra installs the model before evaluation. Disable the upstream import-time
    download so model installation stays under the dependency policy.
    """
    import spacy.cli

    setattr(spacy.cli, "download", lambda *args, **kwargs: None)


# Tokenizer, stopword and POS-tagging resources used by IFBench's scorer and provisioned
# by ``gym init ifbench`` before evaluation.
IFBENCH_NLTK_RESOURCES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


def missing_ifbench_nltk_data() -> list[str]:
    """The names in ``IFBENCH_NLTK_RESOURCES`` that NLTK cannot find."""
    import nltk

    missing = []
    for name, resource in IFBENCH_NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(name)
    return missing


class _LevenshteinForLiveBench(importlib.abc.MetaPathFinder):
    """Once GEPA's LiveBench-Math package is imported, serve ``agents_gym_core._levenshtein``
    as ``Levenshtein`` if that package is absent: its olympiad scorer imports
    ``Levenshtein.distance``. The substitution happens only when that package is imported;
    ``agents_gym_core._levenshtein`` supplies rapidfuzz's distance."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if (
            fullname == "gepa_artifact.benchmarks.livebench_math"
            and "Levenshtein" not in sys.modules
        ):
            find = importlib.util.find_spec
            if find("Levenshtein") is None and find("rapidfuzz"):
                sys.modules["Levenshtein"] = importlib.import_module("agents_gym_core._levenshtein")
        return None


def ensure_gepa_artifact_importable() -> None:
    """Put the checkout on ``sys.path`` when it exists (no-op otherwise), and register the
    ``Levenshtein`` provider for its LiveBench-Math scorer."""
    root = gepa_artifact_root()
    if root.is_dir() and str(root) not in sys.path:
        sys.path.append(str(root))
    if not any(isinstance(finder, _LevenshteinForLiveBench) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LevenshteinForLiveBench())
