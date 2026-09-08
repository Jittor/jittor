"""Build-time preconditions must raise, not log and carry on.

Two ways the old checks could evaporate:

* a bare ``assert`` disappears entirely under ``python -O``, so the check
  simply does not run and the failure surfaces later somewhere unrelated;
* ``LOG.f`` is not a fatal call. ``Logwrapper._log`` starts with
  ``if self.log_silent or verbose > self.log_v: return`` -- with
  ``log_silent=1`` a "fatal" log is a no-op and execution continues past it.
"""

import os
import tempfile
import unittest

import jittor_utils as jit_utils

from _helpers.child_process import run_python_child


class TestSearchFileRaises(unittest.TestCase):

    def test_a_missing_library_raises_with_the_search_path(self):
        from jittor.compile_extern import search_file
        with self.assertRaises(RuntimeError) as caught:
            search_file(["/nonexistent-a", "/nonexistent-b"], "libnothing.so")
        message = str(caught.exception)
        self.assertIn("libnothing.so", message)
        self.assertIn("/nonexistent-a", message)

    def test_it_still_raises_when_logging_is_silenced(self):
        """LOG.f was a no-op under log_silent=1, and the caller ran on."""
        from jittor.compile_extern import search_file
        saved = jit_utils.LOG.log_silent
        jit_utils.LOG.log_silent = 1
        try:
            with self.assertRaises(RuntimeError):
                search_file(["/nonexistent"], "libnothing.so")
        finally:
            jit_utils.LOG.log_silent = saved


def _tool_without_a_version(directory):
    """An executable that answers --version with no number in it."""
    path = os.path.join(directory, "toolchain-thing")
    with open(path, "w") as f:
        f.write("#!/bin/sh\necho 'this build of the tool is unversioned'\n")
    os.chmod(path, 0o755)
    return path


class TestGetVersionRaises(unittest.TestCase):

    def test_output_without_a_version_number_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tool = _tool_without_a_version(d)
            with self.assertRaises(RuntimeError) as caught:
                jit_utils._read_version(tool)
        self.assertIn("version", str(caught.exception))

    def test_the_check_survives_python_O(self):
        """A bare assert would have been compiled out here."""
        directory = tempfile.mkdtemp()
        tool = _tool_without_a_version(directory)
        script = (
            "import jittor_utils as u\n"
            "try:\n"
            "    u._read_version(%r)\n" % tool +
            "except RuntimeError:\n"
            "    print('RAISED')\n"
            "except AssertionError:\n"
            "    print('ASSERTED')\n"
            "else:\n"
            "    print('SILENT')\n")
        try:
            out = run_python_child(["-O", "-c", script], text=False)
        finally:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
        self.assertIn(b"RAISED", out.stdout, out.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
