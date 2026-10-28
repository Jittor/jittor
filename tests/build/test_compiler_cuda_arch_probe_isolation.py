# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA driver query must not re-enter Jittor during a cold import."""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from jittor.build import compiler
from jittor.build.utils import lock


class TestCudaArchProbeIsolation(unittest.TestCase):
    def test_query_child_ignores_importing_startup_hook_while_parent_holds_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, "sitecustomize-ran")
            sitecustomize = os.path.join(directory, "sitecustomize.py")
            with open(sitecustomize, "w") as hook:
                hook.write("import jittor\n")
                hook.write("open(%r, 'w').close()\n" % marker)

            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = os.pathsep.join(
                (directory, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
            )
            child_env["JITTOR_TORCH_SHIM"] = "1"
            real_run = subprocess.run
            calls = []

            def bounded_run(command, **kwargs):
                calls.append(command)
                kwargs["timeout"] = 10
                return real_run(command, **kwargs)

            with mock.patch.object(
                compiler.env_config, "child_env", return_value=child_env
            ), mock.patch.object(compiler.sp, "run", side_effect=bounded_run):
                # Use the actual Jittor lock, so an old child that imports via
                # sitecustomize blocks on the same cache lock the parent owns.
                lock.jittor_lock.lock()
                try:
                    # A missing CUDA driver is a supported probe result on CPU CI;
                    # the regression is that startup must not wait on this lock.
                    result = compiler._read_cuda_archs()
                finally:
                    lock.jittor_lock.unlock()
                startup_hook_ran = os.path.exists(marker)

        self.assertTrue(calls)
        self.assertEqual(calls[0][1:3], ["-I", "-S"])
        self.assertTrue(os.path.isabs(calls[0][3]))
        self.assertEqual(os.path.basename(calls[0][3]), "query_cuda_cc.py")
        self.assertFalse(startup_hook_ran, "child executed inherited sitecustomize")
        if compiler.build_config.has_cuda:
            self.assertTrue(result, "CUDA driver query returned no architectures")
        if result is not None:
            self.assertTrue(all(item.isdigit() for item in result))


if __name__ == "__main__":
    unittest.main()
