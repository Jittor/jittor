"""Import-time and first-use contracts for optional external runtimes."""

from __future__ import print_function

import contextlib
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
    if filename.endswith("/jittor/build/compile_extern.py"):
        calls.append(frame.f_code.co_name)

import jittor_utils as jit_utils
fanouts = []
_inner_run_cmds = jit_utils.run_cmds
def run_cmds(cmds, *a, **kw):
    fanouts.append([kw.get("msg", a[2] if len(a) > 2 else "?"), len(cmds)])
    return _inner_run_cmds(cmds, *a, **kw)
jit_utils.run_cmds = run_cmds

sys.setprofile(profile)
started = time.perf_counter()
import jittor
elapsed = time.perf_counter() - started
sys.setprofile(None)

print("IMPORT_BOOTSTRAP_RESULT " + json.dumps({
    "calls": calls,
    "fanouts": fanouts,
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
        # No assertion on build fan-out here, deliberately. This probe forces
        # `nvcc_path=""`, so it imports a *different* configuration from the
        # session's, and in a CUDA gate that one is usually cold -- it is
        # allowed to build. "A warm import compiles nothing" is asserted where
        # the configuration is known to be warm, in
        # TestCustomOpBuildStamp.test_warm_import_does_not_rebuild_anything.
        #
        # The ceiling below is only here to turn a genuine hang into a
        # failure. It used to be 5 s, which on a box running eight partitions
        # at once was a recurring false red that cost two A/B runs to
        # attribute -- a wall clock is the wrong instrument for this, since
        # what the test is about is `calls` being empty.
        self.assertLess(observed["elapsed"], 120.0)

    def test_first_eligible_cpu_bmm_initializes_mkl(self):
        import jittor as jt
        from jittor.nn.functional import matrix
        from jittor._runtime import backend_libraries

        fake_ops = types.SimpleNamespace(mkl_batched_matmul=object())
        libraries = backend_libraries.BackendLibraries()
        libraries.register_loader("mkl", lambda: jt.compile_extern.setup_mkl(),
                                  enabled=jt.compile_extern._mkl_library_enabled)

        def setup():
            backend_libraries.register_library("mkl", types.SimpleNamespace(ops=fake_ops))

        with jt.flag_scope(use_cuda=0), \
                mock.patch.dict(os.environ, {"use_mkl": "1"}), \
                mock.patch.object(jt.compile_extern, "use_mkl", True), \
                mock.patch.object(backend_libraries, "_libraries", libraries), \
                mock.patch.object(jt.compile_extern, "setup_mkl",
                                  side_effect=setup) as setup_mock:
            operand = jt.array([[1.0]])
            self.assertTrue(
                matrix._mkl_batched_matmul_is_available(operand, operand))
            setup_mock.assert_called_once_with()

    def test_first_cuda_transpose_initializes_cutt(self):
        """cuTT needs a caller, and for a long time it had none.

        9.01 replaced the import-time ``setup_nccl`` / ``setup_cutt`` /
        ``setup_mkl`` calls with lazy ones, but only wrote the lazy call sites
        for NCCL and MKL. ``setup_cutt`` was left with no caller anywhere in
        the tree, which is not a slow path -- it is an unreachable backend:
        ``cutt_ops`` stayed ``None`` forever and every cuTT test skipped with
        a reason that was not true.
        """
        import jittor as jt
        from jittor._core import var

        operand = jt.array([[1.0, 2.0], [3.0, 4.0]])
        with mock.patch.object(var, "_load_accelerator_transpose") as load:
            result = jt.transpose(operand, (1, 0))
        load.assert_called_once_with()
        # The bootstrap wrapper must not change what transpose returns.
        self.assertEqual(result.shape, [2, 2])

    def test_accelerator_transpose_load_is_attempted_once(self):
        from jittor._runtime import backend_libraries
        from jittor._core import var

        requests = []

        def unavailable():
            requests.append("cutt")
            raise RuntimeError("cuTT build failed")

        libraries = backend_libraries.BackendLibraries()
        libraries.register_loader("cutt", unavailable)

        with mock.patch.object(backend_libraries, "_libraries", libraries), \
                mock.patch.object(var, "_accelerator_transpose_tried",
                                  False):
            # A backend that cannot be built is not fatal -- TransposeOp has
            # its own kernel -- but it must not be retried on every transpose.
            var._load_accelerator_transpose()
            var._load_accelerator_transpose()
        self.assertEqual(requests, ["cutt"])

    def test_disabled_mkl_is_not_selected_even_when_already_loaded(self):
        import jittor as jt
        from jittor.nn.functional import matrix
        from jittor._runtime import backend_libraries

        fake_module = types.SimpleNamespace(
            ops=types.SimpleNamespace(mkl_batched_matmul=object()))
        for loaded in (False, True):
            with self.subTest(loaded=loaded):
                libraries = backend_libraries.BackendLibraries()
                libraries.register_loader(
                    "mkl", lambda: jt.compile_extern.setup_mkl(),
                    enabled=jt.compile_extern._mkl_library_enabled)
                if loaded:
                    libraries.register("mkl", fake_module)
                with jt.flag_scope(use_cuda=0), \
                        mock.patch.dict(os.environ, {"use_mkl": "1"}), \
                        mock.patch.object(jt.compile_extern, "use_mkl", False), \
                        mock.patch.object(backend_libraries, "_libraries", libraries), \
                        mock.patch.object(jt.compile_extern, "setup_mkl") as setup_mock:
                    operand = jt.array([[1.0]])
                    self.assertFalse(matrix._mkl_batched_matmul_is_available(operand, operand))
                    setup_mock.assert_not_called()
                    libraries.register("mkl", fake_module)
                    with mock.patch.object(jt.compile_extern, "use_mkl", True):
                        self.assertTrue(matrix._mkl_batched_matmul_is_available(operand, operand))
                    setup_mock.assert_not_called()


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
            source = Path(tree, "src", "ops", "a.cc")
            source.write_text("// one\n")
            # Passed rather than patched: this module's startup configuration
            # is frozen after bootstrap, so `jittor_path` cannot be assigned.
            first = self.compiler.core_source_signature(root=tree)
            self.assertEqual(list(first), [os.path.join("src", "ops", "a.cc")])

            source.write_text("// two\n")
            self.assertEqual(len("// one\n"), len("// two\n"))
            # Set the mtime explicitly: two writes in the same clock tick
            # could otherwise share it, and this test would pass or fail
            # by timing rather than by the property under test.
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            second = self.compiler.core_source_signature(root=tree)
            self.assertNotEqual(first, second)

            Path(tree, "src", "b.h").write_text("")
            third = self.compiler.core_source_signature(root=tree)
            self.assertIn(os.path.join("src", "b.h"), third)
            self.assertNotEqual(second, third)

            comm = Path(tree, "backends", "comm")
            comm.mkdir(parents=True)
            (comm / "__init__.py").write_text("")
            wrapper = comm / "mpi_wrapper.cc"
            wrapper.write_text("// one\n")
            fourth = self.compiler.core_source_signature(root=tree)
            key = os.path.join("backends", "comm", "mpi_wrapper.cc")
            self.assertIn(key, fourth)
            wrapper.write_text("// two\n")
            os.utime(wrapper, ns=(1_000_000_000, 1_000_000_000))
            self.assertNotEqual(fourth, self.compiler.core_source_signature(root=tree))


_CUSTOM_OP_MARKER = "CUSTOM_OP_RESULT "

#: A custom op whose result comes from a header that is *not* in the file list
#: handed to ``compile_custom_ops``.
#:
#: That is the whole point. The op sources are named explicitly, so stat'ing
#: them is easy; the headers they include are found by the per-file dependency
#: scan inside ``compile``, and that scan is exactly what the stamp skips. If
#: the stamp does not cover this header, editing it produces no rebuild and
#: the op keeps returning the old numbers -- a wrong answer, not an error.
_STAMP_HELPER_H = """
#pragma once
#define STAMP_PROBE_FACTOR %d
"""

_STAMP_OP_H = """
#pragma once
#include "op.h"

namespace jittor {

struct StampProbeOp : Op {
    Var* output;
    StampProbeOp(NanoVector shape, NanoString dtype=ns_float32);

    const char* name() const override { return "stamp_probe"; }
    DECLARE_jit_run;
};

} // jittor
"""

_STAMP_OP_CC = """
#include "var.h"
#include "stamp_probe_op.h"
#include "stamp_probe_helper.h"

namespace jittor {
#ifndef JIT
StampProbeOp::StampProbeOp(NanoVector shape, NanoString dtype) {
    output = create_output(shape, dtype);
}

void StampProbeOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", output->dtype());
    add_jit_define(jk, "FACTOR", S(STAMP_PROBE_FACTOR));
}

#else // JIT
void StampProbeOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    for (index_t i=0; i<num; i++)
        x[i] = (T)(i * FACTOR);
}
#endif // JIT

} // jittor
"""

#: Build the op in a child process and print what it computes.
#:
#: A child rather than this process because ``compile_custom_ops`` ends in
#: ``__import__(gen_name)``: the second call in one process gets the module
#: already in ``sys.modules`` and therefore the *first* build's code, however
#: the library on disk was rebuilt. So "did the answer change" is only
#: answerable across processes, and asserting it in-process would pass whether
#: or not the stamp works.
_STAMP_OP_PROBE = r"""
import json, os, sys

import jittor_utils as jit_utils

fanouts = []
_inner = jit_utils.run_cmds
def run_cmds(cmds, *a, **kw):
    fanouts.append([kw.get("msg", a[2] if len(a) > 2 else "?"), len(cmds)])
    return _inner(cmds, *a, **kw)
jit_utils.run_cmds = run_cmds

import jittor as jt

directory = sys.argv[1]
ops = jt.compile_custom_ops([
    os.path.join(directory, "stamp_probe_op.h"),
    os.path.join(directory, "stamp_probe_op.cc"),
])
values = ops.stamp_probe([4], "float32").numpy().tolist()

print("CUSTOM_OP_RESULT " + json.dumps({
    "values": values,
    "built": [name for name, _ in fanouts if "stamp_probe" in name],
    "fanouts": fanouts,
}))
"""


def _write_stamp_op(directory, factor):
    Path(directory, "stamp_probe_helper.h").write_text(
        _STAMP_HELPER_H % factor)
    Path(directory, "stamp_probe_op.h").write_text(_STAMP_OP_H)
    Path(directory, "stamp_probe_op.cc").write_text(_STAMP_OP_CC)


class TestCustomOpBuildStamp(unittest.TestCase):
    """A warm import must not re-check every bundled op library either.

    The core's stamp took the warm CUDA import from 2.46 s to 1.55 s and left
    the same cost in a second place: ``compile_extern`` builds one library per
    bundled CUDA backend, and each one handed its translation units to the
    compile pool so every worker could hash a dependency closure and report
    that there was nothing to do -- ~50 commands and 0.35 s of every import,
    for no work at all.

    The risk is different from the core's, though, because
    ``compile_custom_ops`` is a public API: a stamp that wrongly reports
    "current" does not fail, it silently keeps running the previous build of
    somebody's op. So the tests below are mostly about the ways its inputs can
    change, and the first one is about the input that is easiest to miss.
    """

    def setUp(self):
        import jittor as jt

        self.jt = jt
        self.compiler = jt.compiler

    def _stamped_libraries(self):
        return sorted(Path(self.compiler.cache_path, "custom_ops").glob(
            "*" + self.compiler.extension_suffix + ".build_stamp.json"))

    def test_warm_import_does_not_rebuild_anything(self):
        result = run_python_child(["-c", _CORE_BUILD_PROBE], cwd=_REPO_ROOT,
                                  merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        observed = _probe_result(result.stdout, _CORE_MARKER)
        # Not just the op libraries: every compile fan-out on the import path,
        # so the claim is "a warm import compiles nothing" rather than
        # "a warm import compiles nothing except the one we forgot".
        self.assertEqual(observed["fanouts"], [], observed["fanouts"])
        # Without something like this the assertion above passes vacuously on
        # a configuration that builds no op library at all -- which is every
        # CPU-only configuration, i.e. two of the three gates.
        if self.jt.has_cuda:
            self.assertTrue(
                self._stamped_libraries(),
                "this CUDA configuration carries no custom op stamp, so "
                "nothing above proves the fast path was taken")

    def test_editing_an_unnamed_header_changes_the_answer(self):
        """The silent-wrong-answer case, asserted on the numbers themselves.

        Two child processes, because a rebuilt library cannot replace one this
        process already imported. The first builds with factor 2 and must see
        it; the second builds after the header changed to 3 and must see *3*.
        A stamp blind to this header makes the second child print the first
        child's numbers and exit 0.
        """
        with tempfile.TemporaryDirectory() as directory:
            _write_stamp_op(directory, 2)
            first = run_python_child(["-c", _STAMP_OP_PROBE, directory],
                                     cwd=_REPO_ROOT, merge_stderr=True)
            self.assertEqual(first.returncode, 0, first.stdout[-4000:])
            observed = _probe_result(first.stdout, _CUSTOM_OP_MARKER)
            self.assertEqual(observed["values"], [0.0, 2.0, 4.0, 6.0],
                             observed)
            self.assertTrue(observed["built"],
                            "the first child compiled nothing, so the second "
                            "child proves nothing: " + str(observed))

            _write_stamp_op(directory, 3)
            second = run_python_child(["-c", _STAMP_OP_PROBE, directory],
                                      cwd=_REPO_ROOT, merge_stderr=True)
            self.assertEqual(second.returncode, 0, second.stdout[-4000:])
            observed = _probe_result(second.stdout, _CUSTOM_OP_MARKER)
            self.assertEqual(observed["values"], [0.0, 3.0, 6.0, 9.0],
                             observed)

    def test_an_unchanged_op_is_not_rebuilt(self):
        """The other half: two builds of the same inputs, one compile."""
        with tempfile.TemporaryDirectory() as directory:
            _write_stamp_op(directory, 5)
            first = run_python_child(["-c", _STAMP_OP_PROBE, directory],
                                     cwd=_REPO_ROOT, merge_stderr=True)
            self.assertEqual(first.returncode, 0, first.stdout[-4000:])
            self.assertTrue(_probe_result(first.stdout,
                                          _CUSTOM_OP_MARKER)["built"])

            second = run_python_child(["-c", _STAMP_OP_PROBE, directory],
                                      cwd=_REPO_ROOT, merge_stderr=True)
            self.assertEqual(second.returncode, 0, second.stdout[-4000:])
            observed = _probe_result(second.stdout, _CUSTOM_OP_MARKER)
            self.assertEqual(observed["built"], [], observed["fanouts"])
            self.assertEqual(observed["values"], [0.0, 5.0, 10.0, 15.0],
                             observed)

    def _stamp_inputs(self, library):
        """Read back the inputs recorded beside a built library."""
        with open(self.compiler.product_build_stamp_path(str(library)),
                  encoding="utf8") as handle:
            stamp = json.load(handle)
        return stamp["sources"], stamp["ingredients"]

    def _any_library(self):
        """A built op library that carries a stamp.

        The ``custom_ops`` directory also accumulates libraries whose source
        set no longer exists -- a backend's files get renamed, the generated
        name hashes differently, and the old ``.so`` stays behind forever. So
        the stamps are what to enumerate, not the libraries.
        """
        stamps = self._stamped_libraries()
        if not stamps:
            self.skipTest("this configuration builds no custom op library")
        return str(stamps[0])[:-len(".build_stamp.json")]

    def test_a_current_library_is_recognised(self):
        library = self._any_library()
        sources, ingredients = self._stamp_inputs(library)
        self.assertTrue(self.compiler.product_build_is_current(
            str(library), sources, ingredients))

    def test_every_recorded_input_can_make_the_stamp_stale(self):
        """One subtest per field, so a field that stops mattering is visible.

        A stamp is only as good as the narrowest thing it notices, and the way
        this kind of check rots is that a field quietly stops being compared --
        which no "it works" test can see.
        """
        library = self._any_library()
        sources, ingredients = self._stamp_inputs(library)

        for field in ("files", "includes", "core_sources"):
            with self.subTest(source_field=field):
                changed = dict(sources)
                changed[field] = dict(changed[field])
                changed[field]["jittor-stamp-probe"] = [1, 1]
                self.assertFalse(self.compiler.product_build_is_current(
                    str(library), changed, ingredients))

        for field in ("cc_flags", "extra_flags", "include_flags", "opt_flags",
                      "cc_path", "backend", "nvcc_flags", "version",
                      "extension_suffix"):
            with self.subTest(ingredient=field):
                self.assertIn(field, ingredients)
                changed = dict(ingredients)
                changed[field] = str(changed[field]) + "-stamp-probe"
                self.assertFalse(self.compiler.product_build_is_current(
                    str(library), sources, changed))

        with self.subTest(ingredient="core_output"):
            # The library links -ljittor_core, so a replaced core has to
            # invalidate it even when no source the library names changed.
            changed = dict(ingredients)
            changed["core_output"] = [0, 0]
            self.assertFalse(self.compiler.product_build_is_current(
                str(library), sources, changed))

        with self.subTest(output="replaced"):
            with mock.patch.object(
                    self.compiler, "_stat_signature",
                    return_value={str(library): [0, 0]}):
                self.assertFalse(self.compiler.product_build_is_current(
                    str(library), sources, ingredients))

    def test_a_missing_file_is_recorded_rather_than_skipped(self):
        """"Deleted" and "never listed" must not look the same."""
        with tempfile.TemporaryDirectory() as directory:
            present = os.path.join(directory, "there.h")
            Path(present).write_text("")
            absent = os.path.join(directory, "gone.h")
            signature = self.compiler._stat_signature([present, absent])
            self.assertIsNone(signature[absent])
            self.assertIsNotNone(signature[present])

    def test_the_include_walk_stays_off_the_toolkit_and_the_tree(self):
        """Two exclusions that are the reason this check is affordable.

        The CUDA SDK's include directories arrive through ``extra_flags`` for
        every bundled library. Walking them is ~1400 stats each and cannot
        detect anything the cuda key in ``cache_path`` does not already
        partition on -- six of them cost 90 ms of every warm CUDA import
        before they were excluded. Jittor's own headers are excluded for the
        opposite reason: ``core_source_signature`` already covers them, so
        walking them here would be the same work twice.
        """
        in_tree = os.path.join(self.compiler.jittor_path, "src")
        self.assertEqual(
            self.compiler._include_tree_signature([in_tree]), {})
        # ... and it is genuinely covered by the other half of the record.
        self.assertTrue(any(
            name.startswith("src" + os.sep)
            for name in self.compiler.core_source_signature()))

        for toolkit in self.compiler.cuda_include_dirs:
            with self.subTest(toolkit=toolkit):
                self.assertEqual(
                    self.compiler._include_tree_signature([toolkit]), {})

        # The caller's own directory, which is the case it exists for, is not
        # excluded by either rule.
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "mine.h").write_text("")
            Path(directory, "notes.txt").write_text("")
            walked = self.compiler._include_tree_signature([directory])
            self.assertEqual(list(walked), ["mine.h"])

    def test_include_dirs_include_the_hand_written_minus_i(self):
        """``extra_flags`` is the only way a caller names a tree we cannot guess."""
        dirs = self.compiler._custom_op_include_dirs(
            [], [], ' -I"/quoted/with spaces" -I/bare/path -I  /spaced ')
        self.assertIn("/quoted/with spaces", dirs)
        self.assertIn("/bare/path", dirs)
        self.assertIn("/spaced", dirs)


_GATE_MARKER = "GATE_RESULT "
_GATE_PROBE = r"""
import json

import jittor_utils as jit_utils

fanouts = []
_inner = jit_utils.run_cmds
def run_cmds(cmds, *a, **kw):
    fanouts.append([kw.get("msg", a[2] if len(a) > 2 else "?"), len(cmds)])
    return _inner(cmds, *a, **kw)
jit_utils.run_cmds = run_cmds

error = None
try:
    import jittor as jt
    value = (jt.ones(3) * 2).sum().item()
except BaseException as caught:
    error = [type(caught).__name__, str(caught)]
    value = None

print("GATE_RESULT " + json.dumps({
    "error": error, "value": value, "fanouts": fanouts,
}))
"""


def _run_gate_probe(env, script=_GATE_PROBE, args=()):
    """Run a gate probe, tolerating the documented jit_utils rebuild.

    ``jit_utils_core`` is built before anything in ``jittor`` runs, so it is
    outside the gate: a fresh build configuration rebuilds it and exits asking
    for a rerun (by design, see ``jittor-core-cpp-edit-loop``). That is the
    first run here, not a failure, so do what the message says.
    """
    for _ in range(2):
        result = run_python_child(["-c", script] + list(args),
                                  cwd=_REPO_ROOT, env=dict(env),
                                  merge_stderr=True)
        if "rerun the same command" not in result.stdout:
            return result
    return result


@contextlib.contextmanager
def _looks_unbuilt(compiler):
    """Make this configuration look unbuilt, without building anything.

    The obvious way to get an unbuilt configuration is a distinct
    ``cc_flags``, since that lands in its own cache directory. Three problems,
    all found the hard way: it costs a full cold core build, it leaves a few
    hundred MB behind on a machine that is already at 96% disk, and -- because
    the fingerprint is fixed -- it is only unbuilt the *first* time it is ever
    run, so the test passes once and silently stops testing anything.

    Moving the stamp aside asks the same question of the code under test
    ("what happens when the core is not known to be current") for the price of
    a rename, and it asks it every time.
    """
    path = compiler.core_build_stamp_path()
    hidden = path + ".hidden-by-test"
    os.replace(path, hidden)
    try:
        yield
    finally:
        if os.path.exists(path):
            # A child was allowed to build and left a stamp describing the
            # product that exists now. Keep that one: restoring the old stamp
            # over it would describe a product that may have been relinked.
            os.remove(hidden)
        else:
            os.replace(hidden, path)


class TestNoBuildOnImport(unittest.TestCase):
    """``JITTOR_NO_BUILD=1`` makes "import must not compile" enforceable.

    The plan's finish line for this task is an import that neither builds nor
    downloads. The core's build is still on the import path -- moving it off
    means deferring ``import jittor_core``, and the whole Python layer is built
    on that module object -- so what is available today is the other half of
    the contract: a deployment can *declare* that this import must not build
    and be told, immediately and with the command to run, when it would have.

    Why that matters beyond tidiness: offline and read-only installations are
    where the build does not merely cost forty seconds, it spends them and
    then fails for a reason that has nothing to do with the real problem.
    """

    def setUp(self):
        import jittor as jt

        self.compiler = jt.compiler

    def test_a_warm_cache_imports_and_computes_under_the_gate(self):
        result = _run_gate_probe({"JITTOR_NO_BUILD": "1"})
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        observed = _probe_result(result.stdout, _GATE_MARKER)
        self.assertIsNone(observed["error"], observed)
        self.assertEqual(observed["value"], 6.0, observed)
        self.assertEqual(observed["fanouts"], [], observed)

    def test_an_unbuilt_configuration_fails_closed(self):
        with _looks_unbuilt(self.compiler):
            result = _run_gate_probe({"JITTOR_NO_BUILD": "1"})
        observed = _probe_result(result.stdout, _GATE_MARKER)
        # The probe catches it and exits 0 on purpose, so that "raised the
        # wrong thing" and "raised nothing" are distinguishable here rather
        # than both showing up as a non-zero exit.
        self.assertIsNotNone(observed["error"], observed)
        self.assertEqual(observed["error"][0], "BuildNotAllowed", observed)
        self.assertIsNone(observed["value"], observed)
        # An error is only useful if it says what to do about it.
        self.assertIn("python -m jittor_utils.bootstrap", observed["error"][1])
        # And it has to refuse *before* compiling, not after.
        self.assertEqual(observed["fanouts"], [], observed)

    def test_the_same_state_builds_when_the_gate_is_off(self):
        """The gate must be the reason, not a cache that cannot be built.

        Without this, the test above passes just as well if the hidden stamp
        had left the configuration unusable -- that would also be "no compile
        and no answer", which is what the assertions there look for.
        """
        with _looks_unbuilt(self.compiler):
            result = _run_gate_probe({"JITTOR_NO_BUILD": "0"})
            self.assertEqual(result.returncode, 0, result.stdout[-4000:])
            observed = _probe_result(result.stdout, _GATE_MARKER)
        self.assertIsNone(observed["error"], observed)
        self.assertEqual(observed["value"], 6.0, observed)
        self.assertTrue(observed["fanouts"],
                        "with no stamp, build_core had to re-check the core "
                        "the expensive way, which fans out to the pool")

    def test_the_refusal_is_not_downgraded_to_a_warning(self):
        """``setup_cub`` turns build failures into warnings; not this one.

        That handler exists so a missing cub does not stop an import. A
        refusal to build is not a missing cub, and letting it through the same
        path would produce exactly what the switch exists to prevent: an
        import that "succeeded" with cub silently absent.

        The steps before ``setup_cub`` are stubbed rather than skipped on a
        CPU-only configuration, because the guard under test is not
        CUDA-specific and a test that only runs in one of the three gates
        would not have caught it being removed.
        """
        import jittor as jt

        with mock.patch.object(jt.compile_extern, "has_cuda", True), \
                mock.patch.object(jt.compile_extern, "is_cuda", True), \
                mock.patch.object(jt.compile_extern, "compile_if_stale"), \
                mock.patch.object(jt.compile_extern.ctypes, "CDLL"), \
                mock.patch.object(jt.compile_extern,
                                  "register_library_resources"), \
                mock.patch.object(
                    jt.compile_extern, "setup_cub",
                    side_effect=jt.compiler.BuildNotAllowed("no")) as cub:
            with self.assertRaises(jt.compiler.BuildNotAllowed):
                jt.compile_extern.setup_cuda_extern()
        # Otherwise a setup_cuda_extern that raised earlier for an unrelated
        # reason would satisfy assertRaises and prove nothing.
        cub.assert_called_once_with()

    def test_build_is_allowed_reads_the_switch(self):
        import jittor as jt

        for value, allowed in (("0", True), ("", True), ("1", False),
                               ("yes", False)):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"JITTOR_NO_BUILD": value}):
                    self.assertEqual(jt.compiler.build_is_allowed(), allowed)
        environment = dict(os.environ)
        environment.pop("JITTOR_NO_BUILD", None)
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertTrue(jt.compiler.build_is_allowed())

    def test_an_explicit_build_request_ignores_the_switch(self):
        """``build_core(force=True)`` is the explicit request; it must build."""
        import jittor as jt

        with mock.patch.dict(os.environ, {"JITTOR_NO_BUILD": "1"}):
            with mock.patch.object(jt.compiler, "compile") as compile_mock, \
                    mock.patch.object(jt.compiler, "compile_backend_sources",
                                      return_value=[]), \
                    mock.patch.object(jt.compiler,
                                      "_write_core_build_stamp"):
                jt.compiler.build_core(force=True)
            compile_mock.assert_called_once()


class TestBootstrapEntryPoint(unittest.TestCase):
    """The counterpart that *is* allowed to build.

    It lives in ``jittor_utils`` rather than ``jittor`` because ``python -m
    jittor.bootstrap`` would run the ``jittor`` package body first -- the
    build it is supposed to authorise would already have happened by the time
    it got control.
    """

    def test_bootstrap_reports_a_warm_cache(self):
        result = run_python_child(["-m", "jittor_utils.bootstrap"],
                                  cwd=_REPO_ROOT, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        self.assertIn("bootstrapped in", result.stdout)
        self.assertIn("cache_path:", result.stdout)

    def setUp(self):
        import jittor as jt

        self.compiler = jt.compiler

    def test_bootstrap_builds_even_when_the_gate_is_set(self):
        """The environment that needs bootstrap most is the one that sets it.

        If bootstrap merely inherited ``JITTOR_NO_BUILD=1`` it would refuse to
        do the one job it has, which is the "guard that turns into a no-op"
        shape: the command exists, exits non-zero, and nothing gets built.
        """
        with _looks_unbuilt(self.compiler):
            result = run_python_child(["-m", "jittor_utils.bootstrap"],
                                      cwd=_REPO_ROOT, merge_stderr=True,
                                      env={"JITTOR_NO_BUILD": "1"})
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        self.assertIn("bootstrapped in", result.stdout)

    def test_check_reports_an_unbuilt_cache_without_building_it(self):
        with _looks_unbuilt(self.compiler):
            result = run_python_child(
                ["-m", "jittor_utils.bootstrap", "--check"],
                cwd=_REPO_ROOT, merge_stderr=True)
            self.assertNotEqual(result.returncode, 0, result.stdout[-4000:])
            self.assertIn("BuildNotAllowed", result.stdout)
            self.assertNotIn("Compiling jittor_core", result.stdout)
            # --check must leave the cache exactly as it found it, or the
            # second --check would answer a question about the first one.
            self.assertFalse(
                os.path.isfile(self.compiler.core_build_stamp_path()),
                "--check built something")

    def test_check_passes_on_a_warm_cache(self):
        result = run_python_child(
            ["-m", "jittor_utils.bootstrap", "--check"],
            cwd=_REPO_ROOT, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        self.assertIn("verified in", result.stdout)

    def test_bootstrap_pins_the_child_to_this_checkout(self):
        """The child must build the tree bootstrap belongs to.

        Without the pin it builds whatever ``jittor`` a bare interpreter
        resolves -- in a development checkout, the editable install, i.e. some
        other working tree -- and reports success for a cache the caller will
        never use.
        """
        result = run_python_child(["-m", "jittor_utils.bootstrap"],
                                  cwd=_REPO_ROOT, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:])
        self.assertIn("jittor:     %s" % os.path.dirname(self.compiler.__file__),
                      result.stdout)


if __name__ == "__main__":
    unittest.main()
