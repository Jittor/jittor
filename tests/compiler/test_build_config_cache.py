"""The cache directory must name the build configuration it was built with.

The directory is keyed on the toolchain (compiler, Python, platform, CPU, git
branch) but was not keyed on the *flags* that toolchain is invoked with. The
torch shim appends ``--fmad=false --prec-div=true --prec-sqrt=true`` to
``nvcc_flags`` and drops ``--use_fast_math``, so switching it on and off
produced two different sets of CUDA kernels in one ``jit/`` directory: every
switch recompiled all of them, and two processes running at once replaced each
other's shared libraries -- including ones already dlopen'd.

These tests only need ``jittor_utils``, which computes the path; they never
build anything.
"""

import json
import os
import tempfile
import unittest

import jittor_utils as jit_utils

from _helpers.child_process import run_python_child


def _cache_path_for(env_overrides):
    """cache_path as computed by a fresh interpreter with these variables set."""
    script = ("import jittor_utils, json, sys;"
              "sys.stdout.write(json.dumps(["
              "jittor_utils.cache_path, jittor_utils.lock_path]))")
    out = run_python_child(["-c", script], env=env_overrides, text=False)
    assert out.returncode == 0, out.stderr.decode()
    return json.loads(out.stdout.decode())


class TestBuildConfigFingerprint(unittest.TestCase):

    def test_flags_change_the_fingerprint(self):
        base = jit_utils.build_config_fingerprint({})
        for name in jit_utils.BUILD_CONFIG_VARS:
            changed = jit_utils.build_config_fingerprint({name: "-something"})
            self.assertNotEqual(base, changed,
                                f"{name} does not reach the cache directory")

    def test_unset_and_empty_are_different_configurations(self):
        """`nvcc_path=""` is the documented way to force a CPU-only build.

        Recording an absent variable as "" made it indistinguishable from that,
        so the CPU-only build and the CUDA build shared one directory and
        rebuilt the products they both keep there from under each other.
        """
        self.assertNotEqual(jit_utils.build_config_fingerprint({}),
                            jit_utils.build_config_fingerprint(
                                {"nvcc_path": ""}))

    def test_a_cpu_only_run_gets_its_own_directory(self):
        with_nvcc = _cache_path_for({"nvcc_path": "/usr/local/cuda/bin/nvcc"})
        without = _cache_path_for({"nvcc_path": ""})
        self.assertNotEqual(with_nvcc[0], without[0])
        # ...but they still share one build lock, and so one download area.
        self.assertEqual(with_nvcc[1], without[1])

    def test_fingerprint_is_stable_and_short(self):
        config = {"nvcc_flags": " --fmad=false "}
        first = jit_utils.build_config_fingerprint(config)
        self.assertEqual(first, jit_utils.build_config_fingerprint(dict(config)))
        self.assertTrue(first.startswith("cfg"))
        self.assertEqual(len(first), 11)

    def test_shim_math_flags_get_their_own_cache_directory(self):
        """The exact switch the torch shim flips must not share a directory."""
        plain = _cache_path_for({"nvcc_flags": ""})
        strict = _cache_path_for(
            {"nvcc_flags": " --fmad=false --prec-div=true --prec-sqrt=true "})
        self.assertNotEqual(plain[0], strict[0])

    def test_the_lock_is_shared_across_configurations(self):
        """One lock for the machine's toolchain: it guards shared downloads too."""
        plain = _cache_path_for({"nvcc_flags": ""})
        strict = _cache_path_for({"nvcc_flags": " --fmad=false "})
        self.assertEqual(plain[1], strict[1])
        # ...and it stays above the configuration directory.
        self.assertTrue(plain[0].startswith(os.path.dirname(plain[1])))

    def test_configuration_is_recorded_next_to_the_products(self):
        """cfg1234abcd is unreadable on its own; the knobs sit beside it."""
        record = os.path.join(jit_utils.cache_path, "build_config.json")
        self.assertTrue(os.path.isfile(record), record)
        with open(record) as f:
            saved = json.load(f)
        self.assertEqual(
            jit_utils.build_config_fingerprint(saved),
            os.path.basename(jit_utils.cache_path))


class TestDiskSpaceCheck(unittest.TestCase):
    """A full disk is indistinguishable from a corrupted cache unless we look.

    Out of space, the compiler writes a truncated object file, cache_compile
    records a matching key for it, and the failure surfaces later as scattered
    compile errors and segfaults in unrelated operators.
    """

    class _Stat:
        f_frsize = 1 << 20

        def __init__(self, free_mb):
            self.f_bavail = free_mb

    def _with_free(self, free_mb, **kw):
        real = os.statvfs
        os.statvfs = lambda path: self._Stat(free_mb)
        try:
            return jit_utils.check_cache_disk_space(jit_utils.cache_path, **kw)
        finally:
            os.statvfs = real

    def test_enough_space_is_silent(self):
        self.assertIsNone(self._with_free(50000))

    def test_too_little_space_says_so(self):
        with self.assertRaises(RuntimeError) as caught:
            self._with_free(10)
        message = str(caught.exception)
        self.assertIn("10 MB free", message)
        self.assertIn(jit_utils.cache_path, message)
        self.assertIn("JITTOR_HOME", message)

    def test_the_check_can_be_turned_off(self):
        self.assertIsNone(self._with_free(1, minimum_mb=0))

    def test_a_filesystem_that_will_not_answer_is_not_an_error(self):
        real = os.statvfs

        def refuse(path):
            raise OSError("no")

        os.statvfs = refuse
        try:
            self.assertIsNone(
                jit_utils.check_cache_disk_space(jit_utils.cache_path))
        finally:
            os.statvfs = real


if __name__ == "__main__":
    unittest.main()
