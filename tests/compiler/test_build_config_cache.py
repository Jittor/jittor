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
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import jittor_utils as jit_utils


REPO_PYTHON = str(Path(__file__).resolve().parents[2] / "python")


def _cache_path_for(env_overrides):
    """cache_path as computed by a fresh interpreter with these variables set."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_overrides)
    script = ("import jittor_utils, json, sys;"
              "sys.stdout.write(json.dumps(["
              "jittor_utils.cache_path, jittor_utils.lock_path]))")
    out = subprocess.run([sys.executable, "-c", script], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert out.returncode == 0, out.stderr.decode()
    return json.loads(out.stdout.decode())


class TestBuildConfigFingerprint(unittest.TestCase):

    def test_flags_change_the_fingerprint(self):
        base = jit_utils.build_config_fingerprint({})
        for name in jit_utils.BUILD_CONFIG_VARS:
            changed = jit_utils.build_config_fingerprint({name: "-something"})
            self.assertNotEqual(base, changed,
                                f"{name} does not reach the cache directory")

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


if __name__ == "__main__":
    unittest.main()
