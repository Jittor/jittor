# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A session that died must not read like a session that passed.

A native crash inside a test does not fail that test -- it ends the interpreter.
pytest prints the nodeid it was about to run and the process is gone: no result
line, no traceback, no summary, and every test after it never runs. The log
simply stops. Nothing in it says "failed", so a reader scanning for failures
finds none and a script grepping for them agrees.

That is how the maintained CPU gate was silently running only part of itself: a
`scatter_add` segfault on CPU-only builds (`nvcc_path=""`, which is exactly what
``tools/run_test_suite.py`` configures) took the Torch session out at 48%, and on
machines with CUDA the same session completes -- which is why it went unnoticed.

The detector is ``tools/check_session_completed.py``. This file is what stops it
from becoming decorative: the rule is exercised against a truncated log and a
missing sentinel, because a guard that has never been shown to fire is a guard
nobody has checked.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

pytestmark = pytest.mark.structure

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_session_completed.py"

sys.path.insert(0, str(REPO_ROOT / "tests"))
from _helpers import session_completion  # noqa: E402


TRUNCATED_LOG = """\
tests/ops/test_ops.py::TestCommonCPU::test_reference_gather_float32 PASSED
tests/ops/test_ops.py::TestCommonCPU::test_reference_scatter_add_float32
"""

COMPLETED_LOG = """\
tests/ops/test_ops.py::TestCommonCPU::test_reference_gather_float32 PASSED
========================= 1 passed, 0 failed in 3.20s ==========================
"""


def _run(*args):
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        capture_output=True, text=True)


class TestTheCheckerFires(unittest.TestCase):
    """Both directions, because only one of them is the interesting one."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))

    def test_a_truncated_log_fails(self):
        log = self.tmp / "truncated.log"
        log.write_text(TRUNCATED_LOG, encoding="utf-8")
        result = _run("--log", str(log))
        self.assertEqual(result.returncode, 1, result.stdout)
        # The message has to name where it stopped, or the reader still has to
        # go hunting for the absence this exists to point at.
        self.assertIn("test_reference_scatter_add_float32", result.stdout)

    def test_a_completed_log_passes(self):
        log = self.tmp / "complete.log"
        log.write_text(COMPLETED_LOG, encoding="utf-8")
        result = _run("--log", str(log))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_missing_sentinel_fails(self):
        log = self.tmp / "truncated.log"
        log.write_text(TRUNCATED_LOG, encoding="utf-8")
        result = _run("--log", str(log), "--sentinel", str(self.tmp / "absent.json"))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_a_written_sentinel_passes(self):
        sentinel = self.tmp / "present.json"
        sentinel.write_text(json.dumps({
            "marker": session_completion.MARKER,
            "collected": 12, "executed": 12, "exitstatus": 0}), encoding="utf-8")
        result = _run("--sentinel", str(sentinel))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_finishing_with_too_few_collected_fails(self):
        # The other way a run loses coverage without failing anything: it
        # completes, but over a smaller selection than the gate intends.
        sentinel = self.tmp / "short.json"
        sentinel.write_text(json.dumps({
            "marker": session_completion.MARKER,
            "collected": 40, "executed": 40, "exitstatus": 0}), encoding="utf-8")
        result = _run("--sentinel", str(sentinel), "--expect-collected", "8318")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("collected 40", result.stdout)


class TestThePluginRecordsCompletion(unittest.TestCase):

    def test_the_marker_line_carries_the_counts(self):
        session_completion.record_collected(7)
        line = session_completion.marker_line()
        self.assertIn(session_completion.MARKER, line)
        self.assertIn("collected=7", line)

    def test_the_sentinel_round_trips(self):
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        session_completion.record_collected(3)
        path = session_completion.write_sentinel(tmp / "s.json", {"exitstatus": 0})
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["marker"], session_completion.MARKER)
        self.assertEqual(data["collected"], 3)


if __name__ == "__main__":
    unittest.main()
