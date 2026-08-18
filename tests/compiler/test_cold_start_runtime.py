"""Contracts for a Jittor process that has to build its own runtime.

A cold start compiles ``jit_utils_core`` before anything can import it, then
compiles ``jittor_core`` against it. Getting that order wrong is not a build
failure -- everything imports and computes correctly -- but the process ends up
with **two** mappings of ``jit_utils_core``: the one Python imported, and a
second one the loader pulls in for ``jittor_core``'s dependency after the first
file has been replaced on disk.

Two mappings mean two copies of every flag and of the log-capture buffer defined
in those sources. ``log_capture_scope`` then flips the switch in one copy while
the operators log into the other, and it silently returns nothing. Nine
compiler and tuner test modules assert on captured output, so they all failed
from a cold cache and passed from a warm one.

These tests always start from a throwaway ``JITTOR_HOME`` so they exercise the
cold path rather than whatever the developer's cache happens to hold.
"""

from __future__ import print_function

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_PYTHON = str(Path(__file__).resolve().parents[2] / "python")


_PROBE = r"""
import json, os, sys
import jittor as jt
from jittor import LOG

a = jt.ones((4, 4))
a.data

mapped = sorted({
    line.split()[4]
    for line in open("/proc/self/maps").read().splitlines()
    if "jit_utils_core" in line
})

with jt.log_capture_scope(log_v=1000) as wide:
    LOG.i("python-side")
    (a + a).data

with jt.log_capture_scope(log_v=0, log_vprefix="tuner_manager=100",
                          compile_options={"cold_start_probe": 1}) as tuned:
    (a + a).data

print("PROBE_RESULT " + json.dumps({
    "mapped_inodes": mapped,
    "wide_logs": len(wide),
    "wide_files": sorted({entry["name"] for entry in wide}),
    "tuner_logs": len(tuned),
}))
"""


def _cold_start_probe(home):
    environment = dict(os.environ)
    environment.update({
        "JITTOR_HOME": home,
        "TMPDIR": os.path.join(home, "tmp"),
        "PYTHONPATH": REPO_PYTHON + os.pathsep + environment.get("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": "",
        "nvcc_path": "",
        "use_cuda": "0",
        "use_parallel_op_compiler": "0",
        "JITTOR_TORCH_SHIM": "0",
    })
    os.makedirs(environment["TMPDIR"], exist_ok=True)
    completed = subprocess.run(
        (sys.executable, "-c", _PROBE),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    marker = "PROBE_RESULT "
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError("cold-start probe failed:\n" + completed.stdout[-4000:])


@unittest.skipIf(os.name == "nt", "reads /proc/self/maps")
@unittest.skipUnless(Path("/proc/self/maps").exists(), "needs /proc/self/maps")
class TestColdStartRuntime(unittest.TestCase):
    """Build the runtime from nothing and check the resulting process."""

    @classmethod
    def setUpClass(cls):
        cls._home = tempfile.mkdtemp(prefix="jittor-cold-start-")
        cls.result = _cold_start_probe(cls._home)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._home, ignore_errors=True)

    def test_runtime_library_is_mapped_once(self):
        self.assertEqual(
            len(self.result["mapped_inodes"]),
            1,
            "jit_utils_core is mapped {} times: {}".format(
                len(self.result["mapped_inodes"]), self.result["mapped_inodes"]
            ),
        )

    def test_log_capture_sees_the_core(self):
        # A Python-side message alone proves nothing: that one is emitted through
        # the same module that owns the capture switch. The core's own files are
        # the ones that went missing.
        self.assertGreater(self.result["wide_logs"], 1)
        core_files = [
            name for name in self.result["wide_files"] if name.endswith(".cc")
        ]
        self.assertTrue(
            core_files,
            "captured only {}".format(self.result["wide_files"]),
        )
        self.assertIn("executor.cc", self.result["wide_files"])

    def test_tuner_output_is_captured(self):
        self.assertGreater(
            self.result["tuner_logs"],
            0,
            "log_capture_scope saw no tuner output on a cold cache",
        )


if __name__ == "__main__":
    unittest.main()
