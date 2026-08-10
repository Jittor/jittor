# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from __future__ import print_function

import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest


_JITTOR_MODULES_BEFORE = {
    name for name in sys.modules if name == "jittor" or name.startswith("jittor.")
}
_RUNNER_PATH = os.path.join(os.path.dirname(__file__), "_runner.py")
_SPEC = importlib.util.spec_from_file_location("_jittor_test_runner", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
_JITTOR_MODULES_AFTER = {
    name for name in sys.modules if name == "jittor" or name.startswith("jittor.")
}


class _RecordingLoader(object):
    def __init__(self):
        self.modules = []

    def loadTestsFromName(self, module):
        self.modules.append(module)
        return module


class _RecordingSuite(object):
    def __init__(self):
        self.tests = []

    def addTest(self, test):
        self.tests.append(test)


class TestTestRunner(unittest.TestCase):
    def test_runner_import_does_not_import_jittor(self):
        self.assertEqual(_JITTOR_MODULES_BEFORE, _JITTOR_MODULES_AFTER)

    def test_selection_happens_before_loading(self):
        selected = runner.select_tests(
            [
                "helper.py",
                "test_alpha.py",
                "test_blocked.py",
                "test_only.py",
            ],
            skip_l=1,
            skip_r=3,
            skip_markers=("blocked",),
            test_only={"test_alpha", "test_blocked", "test_only"},
        )
        self.assertEqual(
            [test.name for test in selected],
            ["test_alpha", "test_only"],
        )
        self.assertEqual([test.index for test in selected], [1, 3])

        loader = _RecordingLoader()
        suite = _RecordingSuite()
        runner.load_suite(selected, loader, suite)
        self.assertEqual(
            loader.modules,
            ["jittor.test.test_alpha", "jittor.test.test_only"],
        )
        self.assertEqual(suite.tests, loader.modules)

    def test_empty_skip_value_does_not_skip_everything(self):
        config = runner.test_config_from_env({"test_skip": ""})
        selected = runner.select_tests(
            ["test_alpha.py"],
            skip_markers=config["skip_markers"],
        )
        self.assertEqual([test.name for test in selected], ["test_alpha"])

    def test_main_file_remains_directly_executable(self):
        environ = os.environ.copy()
        environ.update({
            "seperate_test": "0",
            "test_only": "test_that_does_not_exist",
        })
        main_path = os.path.join(os.path.dirname(__file__), "__main__.py")
        result = subprocess.run(
            [sys.executable, main_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environ,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Ran 0 tests", result.stdout)

    def test_separate_failures_and_timeouts_return_nonzero(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 3, stdout=b"failed\n")
            raise subprocess.TimeoutExpired(command, 600, output=b"partial\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "nested", "test.log")
            output = io.StringIO()
            status = runner.run_separate_tests(
                ["tests.failure", "tests.timeout"],
                log_path,
                stream=output,
                run_command=fake_run,
                clock=lambda: 1.0,
            )

            self.assertEqual(status, 1)
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, "r", encoding="utf8") as log_file:
                log_output = log_file.read()

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0][0],
            [sys.executable, "-m", "tests.failure", "-v"],
        )
        self.assertIs(calls[0][1]["shell"], False)
        self.assertIn("FAILED", output.getvalue())
        self.assertIn("TIMEOUT", output.getvalue())
        self.assertEqual(output.getvalue(), log_output)

    def test_result_exit_code_includes_separate_status(self):
        result = unittest.TestResult()
        self.assertEqual(runner.result_exit_code(0, result), 0)
        self.assertEqual(runner.result_exit_code(1, result), 1)

    def test_unexpected_success_returns_nonzero(self):
        result = unittest.TestResult()
        result.unexpectedSuccesses.append(object())
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(runner.result_exit_code(0, result), 1)


if __name__ == "__main__":
    unittest.main()
