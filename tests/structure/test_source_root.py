"""Which checkout a test session imports jittor from is a decision, not a default.

``pyproject.toml`` used to pin it with ``pythonpath = ["python"]``, resolved
against rootdir. Anyone verifying a *copy* of the tree had to remember
``-o pythonpath=<copy>/python`` on every invocation, and forgetting it failed
silently -- the tests passed, against the wrong tree.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

from _helpers import child_process
from _helpers.root_conftest import root_conftest_imports_from_the_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
# The implementation, not the conftest module object. `import conftest` is
# ambiguous in this tree -- see tests/_helpers/root_conftest.py -- and these
# assertions are about what `source_python_dir` *does*, which lives here.
conftest = child_process


class TestSourceRootSelection(unittest.TestCase):

    def test_default_is_the_checkout_holding_the_tests(self):
        os.environ.pop("JITTOR_SOURCE_ROOT", None)
        self.assertEqual(Path(conftest.source_python_dir()),
                         REPO_ROOT / "python")

    def test_environment_variable_selects_another_checkout(self):
        os.environ["JITTOR_SOURCE_ROOT"] = "/somewhere/else"
        try:
            self.assertEqual(Path(conftest.source_python_dir()),
                             Path("/somewhere/else/python"))
        finally:
            os.environ.pop("JITTOR_SOURCE_ROOT", None)

    def test_empty_value_leaves_sys_path_alone(self):
        os.environ["JITTOR_SOURCE_ROOT"] = "  "
        try:
            self.assertIsNone(conftest.source_python_dir())
        finally:
            os.environ.pop("JITTOR_SOURCE_ROOT", None)

    def test_this_session_would_import_jittor_from_this_checkout(self):
        """Located, not imported: importing jittor here would build it."""
        spec = importlib.util.find_spec("jittor")
        self.assertIsNotNone(spec)
        origin = Path(spec.origin).resolve()
        self.assertTrue(
            str(origin).startswith(str(REPO_ROOT / "python")),
            f"pytest would import jittor from {origin}, not from this checkout")


if __name__ == "__main__":
    unittest.main()


class TestOneImplementation(unittest.TestCase):
    """The root conftest must take the function, not write a second one.

    If ``tests/conftest.py`` computed the parent's ``sys.path`` and
    ``_helpers.child_process`` computed the child's ``PYTHONPATH`` separately,
    the two would drift and the drift would be silent: the child would test
    another checkout and every assertion would still pass.
    """

    def test_the_root_conftest_imports_it_from_the_helper(self):
        self.assertTrue(
            root_conftest_imports_from_the_helper("source_python_dir"),
            "tests/conftest.py must take source_python_dir from "
            "_helpers.child_process, not define its own",
        )
