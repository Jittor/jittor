# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One report naming every unmet build precondition, not just the first.

The preconditions used to be checked in whatever order the module-level code
ran in, so a user missing a compiler, Python headers and OpenMP learned about
them one `pip install` at a time -- with a cold build between each.
"""

import os
import tempfile
import unittest
from unittest import mock

from jittor_utils import preflight


class TestPreflightChecks(unittest.TestCase):
    def test_a_missing_compiler_names_what_to_install(self):
        with mock.patch.object(preflight.shutil, "which", return_value=None):
            with mock.patch.dict(os.environ, {"cc_path": ""}, clear=False):
                result = preflight.check_compiler()
        self.assertEqual(result.status, "fail")
        self.assertIn("apt install g++", result.remedy)
        # Installing a compiler is not something Jittor can do for the user,
        # and saying so is the difference between waiting and acting.
        self.assertFalse(result.fixable)

    def test_a_present_compiler_passes(self):
        result = preflight.check_compiler("/usr/bin/g++")
        if not os.path.isfile("/usr/bin/g++"):
            raise unittest.SkipTest("no /usr/bin/g++ on this host")
        self.assertEqual(result.status, "ok")

    def test_python_headers_report_the_interpreter_they_must_match(self):
        result = preflight.check_python_headers()
        if result.status == "ok":
            self.assertTrue(result.detail.endswith("Python.h"))
        else:
            self.assertIn("-dev", result.remedy)

    def test_a_full_disk_is_a_failure_before_the_build_not_after(self):
        """A full disk surfaces as scattered compile failures and segfaults in
        unrelated operators, which reads as a corrupted cache."""
        with tempfile.TemporaryDirectory() as directory:
            result = preflight.check_disk_space(directory, minimum_mb=1 << 40)
        self.assertEqual(result.status, "fail")
        self.assertIn("clean_cache", result.remedy)
        self.assertTrue(result.fixable)

    def test_the_network_is_not_checked_when_nothing_is_missing(self):
        result = preflight.check_network(needed=False)
        self.assertEqual(result.status, "ok")
        self.assertIn("already on disk", result.detail)

    def test_an_unreachable_mirror_points_at_the_offline_route(self):
        result = preflight.check_network(needed=True, host="127.0.0.1",
                                         timeout=0.05)
        self.assertEqual(result.status, "fail")
        self.assertIn("JITTOR_OFFLINE_PATH", result.remedy)
        self.assertTrue(result.fixable)

    def test_an_empty_nvcc_path_is_a_cpu_build_not_a_problem(self):
        result = preflight.check_cuda("")
        self.assertEqual(result.status, "ok")

    def test_every_problem_appears_in_one_report(self):
        results = [
            preflight._fail("a", "a is missing", "install a"),
            preflight._ok("b", "b is fine"),
            preflight._fail("c", "c is missing", "jittor fetches c", True),
        ]
        report = preflight.format_report(results, only_problems=True)
        # Both failures, in one message. This is the whole point.
        self.assertIn("a is missing", report)
        self.assertIn("c is missing", report)
        self.assertNotIn("b is fine", report)
        self.assertIn("needs you", report)
        self.assertIn("Jittor can do this for you", report)
        self.assertEqual(len(preflight.failures(results)), 2)

    def test_assert_ready_raises_once_naming_everything(self):
        broken = [
            preflight._fail("a", "a is missing", "install a"),
            preflight._fail("c", "c is missing", "install c"),
        ]
        with mock.patch.object(preflight, "run_all", return_value=broken):
            with self.assertRaises(RuntimeError) as raised:
                preflight.assert_ready()
        message = str(raised.exception)
        self.assertIn("2 precondition(s)", message)
        self.assertIn("a is missing", message)
        self.assertIn("c is missing", message)

    def test_run_all_covers_the_documented_preconditions(self):
        names = {result.name for result in preflight.run_all()}
        for expected in ("c++ compiler", "python headers", "openmp runtime",
                         "disk space", "third-party archives", "network",
                         "git"):
            self.assertIn(expected, names)

    def test_the_command_line_reports_and_sets_an_exit_status(self):
        with mock.patch.object(preflight, "run_all", return_value=[
                preflight._ok("a", "fine")]):
            self.assertEqual(preflight.main([]), 0)
        with mock.patch.object(preflight, "run_all", return_value=[
                preflight._fail("a", "broken", "fix a")]):
            self.assertEqual(preflight.main([]), 1)
            self.assertEqual(preflight.main(["--json"]), 1)


if __name__ == "__main__":
    unittest.main()
