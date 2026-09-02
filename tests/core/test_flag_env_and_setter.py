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


if __name__ == "__main__":
    unittest.main()
