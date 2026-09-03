# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Flags: environment overrides are parsed whole, and setters run after the write.

Two halves of one contract, both of which used to fail silently.

**Parsing.**  ``get_from_env`` (``utils/log.h``) decided a value had parsed by
reading one more character and checking that *that* failed.  A trailing space
does not fail::

    export log_v="1 "        # -> log_v stays 0, and nothing is said

The one warning it did emit was level ``w``, which ``log_silent`` drops
(``send_log`` in ``log.cc``).  A mistyped flag therefore looked exactly like a
flag that had no effect.  Unparsable values are fatal now.  The same rewrite
fixes ``uint8`` flags, which went through ``operator>>(unsigned char&)`` and
read a *character*: ``node_order=1`` used to mean 49.

``string`` flags do not go through the generic parser at all -- ``log.cc`` has
an explicit ``get_from_env<string>`` specialization that returns the value
verbatim -- so ``gdb_path=""`` and a value with spaces in it keep working.  The
case below pins that down, because a parser that rejected an empty string would
break every crash-isolated child (``child_process._crash_isolated`` clears
``gdb_path``) and every ``nvcc_path=""`` CPU-only run.

**Setter order.**  ``DEFINE_FLAG_WITH_SETTER`` called the setter *before* the
assignment, so every setter saw the old value and each one that needed the new
one assigned it by hand first.  Worse, a setter that *corrected* the requested
value had its correction overwritten by the assignment that followed:
``setter_use_cuda`` falls back to CPU when no device answers, so
``jt.flags.use_cuda = 1`` in a process with no visible GPU printed the fallback
warning and then left ``use_cuda == 1`` anyway.

A child with an unparsable flag dies from ``SIGABRT``: ``LOGf`` throws out of a
static initializer, so it reaches ``std::terminate`` rather than Python.  Those
launches pass ``crash_isolated=True`` -- without it jittor's process-level
``SIGCHLD`` action ends the pytest session itself with no output (6.C31).

Run::  python -m pytest tests/core/test_flag_env_and_setter.py
"""

import unittest

import jittor as jt

from _helpers.child_process import run_child_script


def run_probe(source, extra_env=None, crash_isolated=False):
    done = run_child_script(source, env=extra_env, text=True,
                            merge_stderr=True, name="flag_env",
                            crash_isolated=crash_isolated)
    return done.returncode, done.stdout


PRINT_LOG_V = 'import jittor as jt\nprint("LOG_V", jt.flags.log_v)\nprint("DONE")\n'
PRINT_NODE_ORDER = ('import jittor as jt\nprint("NODE_ORDER", jt.flags.node_order)\n'
                    'print("DONE")\n')
PRINT_STRING_FLAGS = (
    'import jittor as jt\n'
    'print("GDB_PATH[%s]" % jt.flags.gdb_path)\n'
    'print("EXTRA_GDB_CMD[%s]" % jt.flags.extra_gdb_cmd)\n'
    'print("DONE")\n')
FALL_BACK_TO_CPU = (
    'import jittor as jt\n'
    'jt.flags.use_cuda = 1\n'
    'print("USE_CUDA", jt.flags.use_cuda)\n'
    'print("DONE")\n')


class TestFlagEnvParsing(unittest.TestCase):
    def test_a_clean_value_is_used(self):
        code, output = run_probe(PRINT_LOG_V, {"log_v": "1"})
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("LOG_V 1", output, output[-4000:])

    def test_trailing_space_is_reported_not_ignored(self):
        code, output = run_probe(PRINT_LOG_V, {"log_v": "1 "},
                                 crash_isolated=True)
        self.assertNotEqual(code, 0, output[-4000:])
        self.assertIn("log_v", output, output[-4000:])
        self.assertNotIn("LOG_V 0", output, output[-4000:])

    def test_a_bad_value_is_reported_even_when_the_log_is_silent(self):
        # The old warning was level 'w' and log_silent swallowed it, so this
        # combination produced no output at all. "1 " rather than a word,
        # because jittor_utils reads log_v with int() first and int("1 ") is 1
        # -- so this reaches the C++ parser, which is the one under test.
        code, output = run_probe(PRINT_LOG_V,
                                 {"log_v": "1 ", "log_silent": "1"},
                                 crash_isolated=True)
        self.assertNotEqual(code, 0, output[-4000:])
        self.assertIn("log_v", output, output[-4000:])

    def test_a_uint8_flag_is_a_number_not_a_character(self):
        code, output = run_probe(PRINT_NODE_ORDER, {"node_order": "1"})
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("NODE_ORDER 1", output, output[-4000:])
        self.assertNotIn("NODE_ORDER 49", output, output[-4000:])

    def test_a_string_flag_takes_its_value_verbatim(self):
        # Empty and space-containing values are ordinary for string flags:
        # gdb_path="" is how a crash-isolated child asks for no gdb, and
        # extra_gdb_cmd is a command line. Neither may be treated as junk.
        code, output = run_probe(PRINT_STRING_FLAGS,
                                 {"gdb_path": "",
                                  "extra_gdb_cmd": "set pagination off"})
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("GDB_PATH[]", output, output[-4000:])
        self.assertIn("EXTRA_GDB_CMD[set pagination off]", output, output[-4000:])


class TestFlagSetterOrder(unittest.TestCase):
    @unittest.skipIf(not jt.has_cuda, "No cuda found")
    def test_a_setters_correction_is_not_overwritten(self):
        # No device visible: setter_use_cuda warns and falls back to 0. The
        # assignment used to run after the setter and put the requested 1 back.
        code, output = run_probe(FALL_BACK_TO_CPU, {"CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("USE_CUDA 0", output, output[-4000:])

    def test_setting_a_flag_that_had_a_hand_written_write_back_still_works(self):
        # use_cuda_host_allocator's setter used to publish its own value so the
        # get_allocator() call inside it could see it; gdb_path's did the same
        # for the setter it calls. Both hand-written lines are gone, so this
        # round trip is what proves the macro does it now.
        before = jt.flags.use_cuda_host_allocator
        try:
            jt.flags.use_cuda_host_allocator = 0
            self.assertEqual(jt.flags.use_cuda_host_allocator, 0)
            jt.flags.use_cuda_host_allocator = 1
            self.assertEqual(jt.flags.use_cuda_host_allocator, 1)
        finally:
            jt.flags.use_cuda_host_allocator = before

    def test_a_setter_that_throws_leaves_the_flag_alone(self):
        before = jt.flags.log_vprefix
        try:
            with self.assertRaises(Exception):
                jt.flags.log_vprefix = "this is not a prefix spec"
            self.assertEqual(jt.flags.log_vprefix, before)
        finally:
            # The rollback under test is what makes this line a no-op. It is
            # here because the flag-scope contract asks every case to put back
            # what it assigned, and a case about a rollback is the last place
            # that should rely on the rollback working.
            jt.flags.log_vprefix = before


class TestUseCudaLowersOnlyAfterTheFlush(unittest.TestCase):
    """Leaving ``flag_scope(use_cuda=1)`` must flush the graph it built as CUDA.

    ``setter_use_cuda`` calls ``sync_all(0)`` so that the lazy graph standing
    at the switch runs on the backend it was *built* for.  That is not a
    nicety: ``Op::do_jit_prepare`` clears the other backend's flag on every op
    it prepares, so an op prepared under CUDA has ``OpFlags::_cpu`` off
    permanently.  Compiling that graph with ``use_cuda`` already at 0 reaches
    the CPU branch and aborts on ``ASSERT(flag(OpFlags::_cpu))``.

    Before [2.21] the flush got the old value for free, because the macro ran
    the setter before the assignment.  [2.21] put the assignment first -- which
    is what lets a setter correct the value it is handed -- and that silently
    moved this flush to the far side of the switch.  The symptom was a
    ``RuntimeError`` out of ``flag_scope.__exit__``:

        Op broadcast_to doesn't have cpu version

    blamed on whatever line happened to close the scope.  ``tests/ops/
    test_matmul.py::test_backward_cuda`` failed exactly this way, and it left
    ``use_cuda`` at 1 for the rest of the file, so ``test_backward_once`` then
    looked for ``mkl_matmul`` in a process that was still on cuBLAS.

    The graph has to be *held* at the switch for this to bite: the vars below
    stay in scope on purpose.  Dropping them first makes the case pass whether
    or not the bug is present.
    """

    @unittest.skipIf(not jt.has_cuda, "No cuda found")
    def test_leaving_a_cuda_scope_does_not_recompile_its_graph_for_cpu(self):
        import numpy as np

        before = jt.flags.use_cuda
        with jt.flag_scope(use_cuda=1):
            model = jt.nn.Sequential(jt.nn.Linear(1, 10), jt.nn.ReLU(),
                                     jt.nn.Linear(10, 1))
            sgd = jt.nn.SGD(model.parameters(), 0.05, 0.9, 0)
            x = jt.float32(np.random.rand(50, 1))
            y = x * x
            pred_y = model(x)
            loss = (pred_y - y).sqr()
            loss_mean = loss.mean()
            sgd.step(loss_mean)
            # Force the CUDA compile: this is what clears `_cpu` on the ops.
            loss_mean.data.sum()
            # x/y/pred_y/loss/loss_mean stay alive through __exit__ below --
            # that unfinished graph is the thing the flush has to handle.
        self.assertEqual(jt.flags.use_cuda, before)

    @unittest.skipIf(not jt.has_cuda, "No cuda found")
    def test_the_flushed_graph_still_gives_the_right_answer(self):
        # The flush is not just "must not raise": it has to actually produce
        # the values, and produce them once.  A fix that skipped the flush
        # would pass the case above and lose the graph here.
        import numpy as np

        a = np.random.rand(32, 16).astype("float32")
        b = np.random.rand(16, 8).astype("float32")
        with jt.flag_scope(use_cuda=1):
            va, vb = jt.array(a), jt.array(b)
            warm = jt.matmul(va, vb)
            warm.sync()
            held = jt.matmul(va, vb) + warm
        np.testing.assert_allclose(held.data, np.matmul(a, b) * 2,
                                   rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
