"""Toolchain probes must be answered from disk, not by starting processes.

A warm ``import jittor`` used to run ``g++ --version``, ``git branch``,
``python3-config`` twice, ``nvcc --version`` six times, ``gdb --version``,
``mpicc --showme`` three times and a whole second interpreter to read the
GPUs' compute capabilities -- every time, in every process, including every
DataLoader worker and every gate subprocess.
"""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import jittor_utils as jit_utils
from jittor_utils import probe

from _helpers.child_process import run_python_child


class TestProbeCache(unittest.TestCase):

    def setUp(self):
        self.calls = []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for name, value in (("cache_file", lambda: os.path.join(directory.name, "probe.json")),
                            ("_entries", None)):
            patch = mock.patch.object(probe, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def _compute(self, value):
        def run():
            self.calls.append(value)
            return value
        return run

    def test_answer_is_reused_until_the_tool_changes(self):
        with tempfile.TemporaryDirectory() as d:
            tool = os.path.join(d, "tool")
            with open(tool, "w") as f:
                f.write("v1")
            key = "test:" + tool
            self.assertEqual(probe.cached(key, [tool], self._compute("a")), "a")
            self.assertEqual(probe.cached(key, [tool], self._compute("b")), "a")
            self.assertEqual(self.calls, ["a"], "the second call re-probed")

            # Same size, different content: mtime is what moves.
            time.sleep(0.01)
            with open(tool, "w") as f:
                f.write("v2")
            self.assertEqual(probe.cached(key, [tool], self._compute("c")), "c")

    def test_a_tool_that_cannot_report_a_version_is_remembered_as_failing(self):
        """A broken tool on PATH must not be re-run on every import."""
        with tempfile.TemporaryDirectory() as d:
            tool = os.path.join(d, "broken-tool")
            with open(tool, "w") as f:
                f.write("#!/bin/sh\nexit 1\n")
            os.chmod(tool, 0o755)
            before = probe.MISSES
            with self.assertRaises(RuntimeError):
                jit_utils.get_version(tool)
            self.assertEqual(probe.MISSES, before + 1)
            with self.assertRaises(RuntimeError):
                jit_utils.get_version(tool)
            self.assertEqual(probe.MISSES, before + 1, "the failure was probed again")
            # Repairing the tool changes its file, and that is what re-probes.
            time.sleep(0.01)
            with open(tool, "w") as f:
                f.write("#!/bin/sh\necho 'tool 1.2.3'\n")
            self.assertEqual(jit_utils.get_version(tool), "(1.2.3)")
            self.assertEqual(probe.MISSES, before + 2)

    def test_a_missing_tool_is_a_state_of_its_own(self):
        with tempfile.TemporaryDirectory() as d:
            tool = os.path.join(d, "later")
            key = "test-missing:" + tool
            self.assertEqual(probe.cached(key, [tool], self._compute("no")), "no")
            with open(tool, "w") as f:
                f.write("now here")
            self.assertEqual(probe.cached(key, [tool], self._compute("yes")), "yes")

    def test_extra_key_invalidates(self):
        key = "test-extra"
        self.assertEqual(probe.cached(key, [], self._compute("d1"), extra="r1"), "d1")
        self.assertEqual(probe.cached(key, [], self._compute("d2"), extra="r2"), "d2")

    def test_the_cache_survives_a_damaged_file(self):
        """A truncated probe.json must not stop the process from starting."""
        with open(probe.cache_file(), "w") as f:
            f.write("{not json")
        probe.forget.__doc__  # keep the reference explicit
        probe._entries = None
        self.assertEqual(probe.cached("test-damaged", [], self._compute("ok")), "ok")


class TestWarmImportRunsNoProbes(unittest.TestCase):
    """The acceptance criterion: a warm import spawns no probe subprocess."""

    def _import_and_report(self, cache_path):
        script = ("import sys,importlib.util;"
                  "spec=importlib.util.spec_from_file_location('jittor.build.utils.probe',sys.argv[2]);"
                  "probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe);"
                  "probe.cache_file=lambda: sys.argv[1];"
                  "sys.modules[spec.name]=probe;sys.modules['jittor_utils.probe']=probe;"
                  "import jittor;print('MISSES', probe.MISSES)")
        source = Path(__file__).resolve().parents[2] / "python/jittor/build/utils/probe.py"
        out = run_python_child(["-c", script, cache_path, str(source)], text=False)
        assert out.returncode == 0, out.stderr.decode()[-4000:]
        for line in out.stdout.decode().splitlines():
            if line.startswith("MISSES "):
                return int(line.split()[1])
        raise AssertionError("child did not report: " + out.stdout.decode()[-2000:])

    def test_second_import_probes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "probe.json")
            first = self._import_and_report(cache_path)
            self.assertGreater(first, 0, "nothing was probed even from an empty cache")
            self.assertEqual(self._import_and_report(cache_path), 0,
                             "a warm import is still starting probe subprocesses")


if __name__ == "__main__":
    unittest.main()
