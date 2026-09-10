# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""An error message has to contain the reason, and put it first.

Three properties, each one asserted because it was measured to be false.

**The reason has to be in the exception.** A one-symbol typo in a `jt.code`
source used to raise a 3059-character message that was almost entirely `-I` and
`-L` flags: the compiler's own `error: ... was not declared in this scope`
appeared in it **zero** times. It went to stderr and nowhere else, so anyone who
caught the exception, logged it, or ran under a harness that captures output
instead of a terminal got a message with no cause in it at all.

**The reason has to come before the machinery.** An out-of-range index reported
the JIT source path, the op type, the inputs, the outputs and the backtrace
first; the sentence the reader needed was the eighth line of twelve.

**Advice has to be given once.** The "rerun with `JT_SYNC=1`" instruction was
printed inside the report *and* again in the banner that follows it.

What is asserted here is position and presence, never the exact wording. A test
that pinned the phrasing would go red on every improvement to it, which is how
a message ends up frozen at its first draft.
"""

import unittest

import numpy as np

import jittor as jt


#: A source that cannot compile, for a reason the compiler states in one line.
BROKEN_KERNEL = "out0_p[0] = this_symbol_does_not_exist;"


class TestCompileFailureCarriesTheDiagnostic(unittest.TestCase):

    def _failure_text(self):
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(Exception) as caught:
                jt.code([1], "float32", [jt.ones(1)],
                        cpu_src=BROKEN_KERNEL).sync()
        return str(caught.exception)

    def test_the_compiler_diagnostic_is_in_the_exception(self):
        text = self._failure_text()
        self.assertIn(
            "this_symbol_does_not_exist", text,
            "the compiler said what was wrong and the exception does not "
            "repeat it; a caller who logs the exception learns nothing")
        self.assertIn("error:", text)

    def test_the_command_line_is_not_in_the_exception(self):
        """The flag soup is reproducible from the build config; the error is not.

        `-I` and `-L` groups are the bulk of what the old message was. They are
        identical on every failure, so they carry no information about this
        one, and they pushed the diagnostic past where anyone reads.
        """
        text = self._failure_text()
        flag_groups = text.count('-I"') + text.count("-I'") + text.count(" -I/")
        flag_groups += text.count('-L"') + text.count("-L'") + text.count(" -L/")
        self.assertEqual(
            flag_groups, 0,
            "the compile command's include/library flags are back in the "
            "exception text:\n" + text[:400])

    def test_it_says_how_to_get_the_full_command(self):
        """Removing something is only right if it stays reachable."""
        self.assertIn("log_v", self._failure_text())


class TestOperatorFailurePutsTheReasonFirst(unittest.TestCase):

    def _failure_text(self):
        with jt.flag_scope(use_cuda=0):
            source = jt.array(np.arange(5, dtype="float32"))
            with self.assertRaises(RuntimeError) as caught:
                source[jt.array([99])].sync()
        return str(caught.exception)

    def test_the_reason_precedes_the_op_details(self):
        text = self._failure_text()
        reason = text.find("out of bounds")
        self.assertNotEqual(reason, -1, "the reason is missing entirely:\n" + text)
        for later in ("jit source:", "\nop:", "in:"):
            with self.subTest(section=later.strip()):
                where = text.find(later)
                if where == -1:
                    continue
                self.assertLess(
                    reason, where,
                    "%r comes before the reason; the reader has to scroll past "
                    "the machinery to reach the sentence they need:\n%s"
                    % (later.strip(), text))

    def test_the_reason_is_near_the_top(self):
        """Position, measured in lines, because that is what a reader pays."""
        lines = self._failure_text().splitlines()
        index = next((i for i, line in enumerate(lines)
                      if "out of bounds" in line), None)
        self.assertIsNotNone(index, "the reason is missing entirely")
        self.assertLessEqual(
            index, 4,
            "the reason is on line %d of %d; it used to be line 8 of 12, which "
            "is the defect this asserts against" % (index + 1, len(lines)))

    def test_the_rerun_advice_appears_at_most_once(self):
        text = self._failure_text()
        self.assertLessEqual(
            text.count("JT_SYNC"), 1,
            "the same instruction is given more than once:\n" + text)

    def test_no_log_line_prefix_at_all(self):
        """An exception carries no timestamp, thread id or level letter.

        A log *line* opens with `[f <timestamp> <thread> <file:line>]` because
        it is written for a terminal. An exception is read in a Python
        traceback, a log file or a bug report, where the timestamp and thread
        say nothing -- and it used to embed a whole *second* such prefix inside
        the outer one, two timestamps and two source locations for one error.

        The `file:line` survives, because it names where the check lives; it is
        rewritten as an ordinary `binary_op.cc:426:` so it reads as a location
        rather than as log furniture.
        """
        text = self._failure_text()
        self.assertEqual(
            text.count("[f "), 0,
            "a log line prefix reached an exception message:\n" + text)
        self.assertIn("getitem_op.cc:", text,
                      "the source location of the failing check was dropped")

    def test_a_synchronous_check_is_prefix_free_too(self):
        """Not only the async path.

        The async operator report had its inner prefix stripped first; every
        ordinary shape or dtype check went through a different throw site and
        kept its own. Those are the common case -- a shape mismatch is what
        most people hit -- so the stripping belongs at the throw, not in one
        reporter.
        """
        with self.assertRaises(RuntimeError) as caught:
            (jt.ones((3, 4)) + jt.ones((5, 6))).sync()
        text = str(caught.exception)
        self.assertEqual(text.count("[f "), 0,
                         "synchronous check kept its log prefix:\n" + text)
        self.assertIn("binary_op.cc:", text,
                      "the source location was dropped")
        self.assertIn("Shape not match", text,
                      "the reason is not in the message:\n" + text)


if __name__ == "__main__":
    unittest.main()
