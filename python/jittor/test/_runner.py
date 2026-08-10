# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

from __future__ import print_function

from collections import namedtuple
import os
import subprocess
import sys
import time


DEFAULT_TEST_TIMEOUT = 60 * 10
SelectedTest = namedtuple("SelectedTest", ("index", "name", "module"))


def test_config_from_env(environ):
    skip_markers = tuple(
        marker for marker in environ.get("test_skip", "").split(",") if marker
    )
    test_only = None
    if "test_only" in environ:
        test_only = set(environ["test_only"].split(","))

    return {
        "skip_l": int(environ.get("test_skip_l", "0")),
        "skip_r": int(environ.get("test_skip_r", "1000000")),
        "skip_markers": skip_markers,
        "test_only": test_only,
        # Keep the historical misspelling as part of the public CLI contract.
        "separate": environ.get("seperate_test", "1") == "1",
    }


def select_tests(test_files, skip_l=0, skip_r=1000000,
                 skip_markers=(), test_only=None):
    selected = []
    for index, test_file in enumerate(sorted(test_files)):
        if not test_file.startswith("test_"):
            continue

        test_name = test_file.split(".")[0]
        if index < skip_l or index > skip_r:
            continue
        if test_only is not None and test_name not in test_only:
            continue
        if any(marker in test_name for marker in skip_markers):
            continue

        selected.append(SelectedTest(
            index=index,
            name=test_name,
            module="jittor.test." + test_name,
        ))
    return selected


def load_suite(selected_tests, loader, suite):
    for selected in selected_tests:
        suite.addTest(loader.loadTestsFromName(selected.module))
    return suite


def _as_text(output):
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf8", errors="replace")
    return output


def run_separate_tests(test_modules, log_path, timeout=DEFAULT_TEST_TIMEOUT,
                       executable=None, stream=None, run_command=None,
                       clock=None):
    executable = executable or sys.executable
    stream = stream or sys.stdout
    run_command = run_command or subprocess.run
    clock = clock or time.time
    start = clock()
    failures = []

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf8") as log_file:
        def emit(message):
            stream.write(message)
            log_file.write(message)
            log_file.flush()

        for index, test_module in enumerate(test_modules):
            progress = "%d/%d" % (index, len(test_modules))
            emit("[RUN TEST %s] %s\n" % (progress, test_module))
            command = [executable, "-m", test_module, "-v"]
            status = "OK"
            failed = False
            output = ""

            try:
                result = run_command(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    shell=False,
                )
                output = _as_text(result.stdout)
                failed = result.returncode != 0
                if failed:
                    status = "FAILED"
            except subprocess.TimeoutExpired as error:
                output = _as_text(error.output)
                failed = True
                status = "TIMEOUT"
            except OSError as error:
                output = "%s\n" % error
                failed = True
                status = "FAILED"

            emit(output)
            message = "[RUN TEST %s %s] %s %.1f\n" % (
                progress,
                status,
                test_module,
                clock() - start,
            )
            emit(message)
            if failed:
                failures.append(message)

        if failures:
            emit("".join(failures))

    return 1 if failures else 0


def result_exit_code(separate_status, result):
    return 0 if not separate_status and result.wasSuccessful() else 1
