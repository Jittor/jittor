"""One source of repository test roots, independent of any runtime import."""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"
TEST_ROOTS = (TEST_ROOT, REPO_ROOT / "compat/tests", REPO_ROOT / "adapters/tests")


def iter_test_files(pattern="test_*.py"):
    return sorted(
        path
        for root in TEST_ROOTS
        for path in root.rglob(pattern)
        if "__pycache__" not in path.parts
    )


def relative_test_path(path):
    """A Path relative to native tests; compat paths retain their ../ prefix."""
    return Path(os.path.relpath(path, TEST_ROOT))
