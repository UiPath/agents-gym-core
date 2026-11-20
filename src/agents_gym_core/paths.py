"""Where the agents are, and where runs write, in a source checkout and in an installed wheel.

In a source checkout the package is ``<repo>/src/agents_gym_core`` and runs write under
``<repo>`` (``experiments/``, the SDK entry points). An installed wheel has no repo, so runs
write under the working directory instead. The agents always ship inside the package.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = PACKAGE_DIR / "agents"


def source_checkout() -> Path | None:
    """The repository this package runs from, or ``None`` for an installed wheel."""
    repo = PACKAGE_DIR.parents[1]
    is_checkout = PACKAGE_DIR.parent.name == "src" and (repo / "pyproject.toml").is_file()
    return repo if is_checkout else None


def project_root() -> Path:
    """The source checkout, else the working directory."""
    return source_checkout() or Path.cwd()
