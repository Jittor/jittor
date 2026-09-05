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
    "cupy_loaded": "cupy" in sys.modules,
}))
"""


_CORE_MARKER = "CORE_BUILD_RESULT "
_CORE_BUILD_PROBE = r"""
import json
import os

import jittor_utils

fanouts = []
_inner = jittor_utils.run_cmds


def run_cmds(cmds, *args, **kw):
    # Every build in the import path goes through here, so recording the
    # calls answers "did importing jittor compile anything" without having
    # to trust a log line or a wall-clock number.
    fanouts.append([kw.get("msg", args[2] if len(args) > 2 else "?"),
                    len(cmds)])
    return _inner(cmds, *args, **kw)


jittor_utils.run_cmds = run_cmds

import jittor

print("CORE_BUILD_RESULT " + json.dumps({
    "fanouts": fanouts,
    "rebuilt_again": jittor.compiler.build_core(),
    "files": len(jittor.compiler.files),
    "stamp": os.path.isfile(jittor.compiler.core_build_stamp_path()),
}))
"""


def _probe_result(output, marker=_MARKER):
    for line in output.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
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
        self.assertFalse(observed["cupy_loaded"], observed)
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


class TestCoreBuildStamp(unittest.TestCase):
    """``import jittor`` must not rebuild a core that is already built.

    Finding that out the expensive way -- regenerate the headers, then hand
    every core translation unit to the compile pool so each worker can hash a
    dependency closure and report "nothing to do" -- was 0.9 s of every warm
    import, two thirds of the total. The stamp is what makes the answer cheap;
    these tests pin both halves of it: that a current build is recognised, and
    that every kind of change to its inputs is not mistaken for one.
    """

    def setUp(self):
        # Importing here rather than at module scope guarantees a stamp exists
        # for *this* configuration before anything asserts about it, whatever
        # state the cache was in when the session started.
        import jittor as jt

        self.compiler = jt.compiler

    def test_warm_import_does_not_compile_the_core(self):
        result = run_python_child(["-c", _CORE_BUILD_PROBE], cwd=_REPO_ROOT,
                                  merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        observed = _probe_result(result.stdout, _CORE_MARKER)
        self.assertTrue(observed["stamp"], observed)
        core = [name for name, _ in observed["fanouts"]
                if "jittor_core" in name]
        self.assertEqual(core, [], observed["fanouts"])
        # Calling the entry point again must also find nothing to do: the
        # stamp has to be self-consistent with what the build just wrote, or
        # every second import pays for the first one's work again.
        self.assertFalse(observed["rebuilt_again"], observed)
        self.assertGreater(observed["files"], 100, observed)

    def test_stamp_records_the_compile_order(self):
        with open(self.compiler.core_build_stamp_path(),
                  encoding="utf8") as handle:
            stamp = json.load(handle)
        self.assertEqual(stamp["files"], self.compiler.files)
        self.assertGreater(len(self.compiler.files), 100)
        # Written to a temporary name and renamed, so a reader never sees a
        # half-written stamp -- and no temporary is left behind.
        leftovers = list(Path(self.compiler.cache_path).glob(
            os.path.basename(self.compiler.core_build_stamp_path())
            + ".tmp.*"))
        self.assertEqual(leftovers, [])

    def test_an_edited_core_source_makes_the_stamp_stale(self):
        signature = self.compiler.core_source_signature()
        self.assertTrue(
            self.compiler.core_build_is_current(signature=signature))
        name = os.path.join("src", "executor.cc")
        self.assertIn(name, signature)
        edited = dict(signature)
        edited[name] = [signature[name][0] + 1, signature[name][1]]
        self.assertFalse(self.compiler.core_build_is_current(signature=edited))

    def test_a_changed_compile_flag_makes_the_stamp_stale(self):
        self.assertTrue(self.compiler.core_build_is_current())
        with mock.patch.dict(
                os.environ,
                {"nvcc_flags": os.environ.get("nvcc_flags", "")
                 + " -DJITTOR_CORE_STAMP_PROBE"}):
            self.assertFalse(self.compiler.core_build_is_current())

    def test_a_changed_generator_makes_the_stamp_stale(self):
        """The stamp must cover Python code that writes generated C++ files."""
        self.assertTrue(self.compiler.core_build_is_current())
        current = self.compiler.core_generator_signature()
        changed = dict(current)
        changed["files"] = dict(current["files"])
        compiler_name = os.path.relpath(self.compiler.__file__,
                                        self.compiler.jittor_path)
        changed["files"][compiler_name] = dict(
            changed["files"][compiler_name])
        changed["files"][compiler_name]["sha256"] = "0" * 64
        with mock.patch.object(self.compiler, "core_generator_signature",
                               return_value=changed):
            self.assertFalse(self.compiler.core_build_is_current())

    def test_a_replaced_core_library_makes_the_stamp_stale(self):
        self.assertTrue(self.compiler.core_build_is_current())
        with mock.patch.object(self.compiler, "_core_output_signature",
                               return_value=[0, 0]):
            self.assertFalse(self.compiler.core_build_is_current())

    def test_source_signature_sees_same_size_edits_and_new_files(self):
        """The two changes a recorded dependency list cannot see on its own.

        A same-size edit is why the signature carries the nanosecond mtime and
        not just the size; a brand new file is why it is a walk of the source
        tree rather than the dependency lists the last build wrote, which can
        only name files that already existed then.
        """
        with tempfile.TemporaryDirectory() as tree:
            os.makedirs(os.path.join(tree, "src", "ops"))
            os.makedirs(os.path.join(tree, "extern"))
            source = Path(tree, "src", "ops", "a.cc")
            source.write_text("// one\n")
            with mock.patch.object(self.compiler, "jittor_path", tree):
                first = self.compiler.core_source_signature()
                self.assertEqual(list(first), [os.path.join("src", "ops",
                                                            "a.cc")])

                source.write_text("// two\n")
                self.assertEqual(len("// one\n"), len("// two\n"))
                # Set the mtime explicitly: two writes in the same clock tick
                # could otherwise share it, and this test would pass or fail
                # by timing rather than by the property under test.
                os.utime(source, ns=(1_000_000_000, 1_000_000_000))
                second = self.compiler.core_source_signature()
                self.assertNotEqual(first, second)

                Path(tree, "src", "b.h").write_text("")
                third = self.compiler.core_source_signature()
                self.assertIn(os.path.join("src", "b.h"), third)
                self.assertNotEqual(second, third)


if __name__ == "__main__":
    unittest.main()
