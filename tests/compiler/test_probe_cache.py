"""Toolchain probes must be answered from disk, not by starting processes.

A warm ``import jittor`` used to run ``g++ --version``, ``git branch``,
``python3-config`` twice, ``nvcc --version`` six times, ``gdb --version``,
``mpicc --showme`` three times and a whole second interpreter to read the
GPUs' compute capabilities -- every time, in every process, including every
DataLoader worker and every gate subprocess.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import jittor_utils as jit_utils
from jittor_utils import probe


REPO_PYTHON = str(Path(__file__).resolve().parents[2] / "python")


class TestProbeCache(unittest.TestCase):

    def setUp(self):
        self.calls = []

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

    def _import_and_report(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
        script = ("import jittor;"
                  "from jittor_utils import probe;"
                  "print('MISSES', probe.MISSES)")
        out = subprocess.run([sys.executable, "-c", script], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert out.returncode == 0, out.stderr.decode()[-4000:]
        for line in out.stdout.decode().splitlines():
            if line.startswith("MISSES "):
                return int(line.split()[1])
        raise AssertionError("child did not report: " + out.stdout.decode()[-2000:])

    def test_second_import_probes_nothing(self):
        probe.forget()
        first = self._import_and_report()
        self.assertGreater(first, 0, "nothing was probed even from an empty cache")
        self.assertEqual(self._import_and_report(), 0,
                         "a warm import is still starting probe subprocesses")


if __name__ == "__main__":
    unittest.main()
