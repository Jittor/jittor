"""Import-time and first-use contracts for optional external runtimes."""

from __future__ import print_function

import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from _helpers.child_process import run_python_child


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MARKER = "IMPORT_BOOTSTRAP_RESULT "
_IMPORT_PROBE = r"""
import json
import sys
import time

calls = []
names = {"setup_nccl", "setup_cutt", "setup_mkl"}

def profile(frame, event, arg):
    if event != "call" or frame.f_code.co_name not in names:
        return
    filename = frame.f_code.co_filename.replace("\\", "/")
    if filename.endswith("/jittor/compile_extern.py"):
        calls.append(frame.f_code.co_name)

sys.setprofile(profile)
started = time.perf_counter()
import jittor
elapsed = time.perf_counter() - started
sys.setprofile(None)

print("IMPORT_BOOTSTRAP_RESULT " + json.dumps({
    "calls": calls,
    "elapsed": elapsed,
    "setups": [
        callable(getattr(jittor.compile_extern, name, None))
        for name in sorted(names)
    ],
}))
"""


def _probe_result(output):
    for line in output.splitlines():
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    raise AssertionError("import probe produced no result:\n" + output[-4000:])


class TestImportBootstrapLaziness(unittest.TestCase):

    def test_plain_import_does_not_call_external_setups(self):
        with tempfile.TemporaryDirectory() as readonly_home:
            os.chmod(readonly_home, 0o555)
            result = run_python_child(
                ["-c", _IMPORT_PROBE],
                cwd=_REPO_ROOT,
                env={
                    "HOME": readonly_home,
                    "XDG_CACHE_HOME": readonly_home,
                    "JITTOR_OFFLINE_PATH": readonly_home,
                    "CUDA_VISIBLE_DEVICES": "",
                    "nvcc_path": "",
                    "http_proxy": "http://127.0.0.1:9",
                    "https_proxy": "http://127.0.0.1:9",
                },
                without_torch_mode=True,
                merge_stderr=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        observed = _probe_result(result.stdout)
        self.assertEqual(observed["calls"], [])
        self.assertEqual(observed["setups"], [True, True, True])
        # This is a regression ceiling, not the plan's <1 s finish line. The
        # remaining core bootstrap is measured and kept visible on the board.
        self.assertLess(observed["elapsed"], 5.0)

    def test_first_eligible_cpu_bmm_initializes_mkl(self):
        import jittor as jt
        from jittor.nn.functional import matrix

        fake_ops = types.SimpleNamespace(mkl_batched_matmul=object())

        def setup():
            jt.compile_extern.mkl_ops = fake_ops

        operand = types.SimpleNamespace(dtype=jt.float32)
        with jt.flag_scope(use_cuda=0), \
                mock.patch.object(jt.compile_extern, "mkl_ops", None), \
                mock.patch.object(jt.compile_extern, "setup_mkl",
                                  side_effect=setup) as setup_mock:
            self.assertTrue(
                matrix._mkl_batched_matmul_is_available(operand, operand))
            setup_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
