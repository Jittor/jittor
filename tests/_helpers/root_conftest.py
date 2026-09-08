"""Read the canonical pytest policy without depending on conftest module names.

The three conftest adapters register one plugin. Source-contract tests inspect
that implementation directly, regardless of which test root loaded it first.
The public helper spellings remain stable for existing structural tests.
"""

from pathlib import Path


ROOT_CONFTEST = Path(__file__).resolve().parent / "pytest_policy.py"


def root_conftest_source():
    return ROOT_CONFTEST.read_text(encoding="utf-8")


def root_conftest_imports_from_the_helper(name):
    """Whether the shared pytest policy takes ``name`` from ``_helpers.child_process``."""
    return ("from _helpers.child_process import %s" % name) in root_conftest_source()
