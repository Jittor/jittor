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

**Advice has to be given once, and has to work.** The "rerun with `JT_SYNC=1`"
instruction was printed inside the report *and* again in the banner that
follows it -- and `JT_SYNC` is a build macro, so following it rebuilt the core
and exited with status 3 without producing a backtrace.

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
        self.assertIn("JT_LOG_V=1", self._failure_text())

    def test_the_location_is_in_the_users_source(self):
        """The source is spliced into code_op.cc's template; the line is the user's.

        It was reported as `code_op.cc:456:28`, a line of the framework, with
        `@`-macro expansions and inserted defines shifting it further.
        """
        source = "\n    @PRECALC\n    int k = 0;\n    out0_p[k] = not_a_symbol;\n"
        header = "#include <cmath>\nint broken_header = ;"
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(Exception) as caught:
                jt.code([1], "float32", [jt.ones(1)], cpu_src=source).sync()
            self.assertIn("jt.code source:4:", str(caught.exception))
            self.assertNotIn("code_op.cc", str(caught.exception))
            with self.assertRaises(Exception) as caught:
                jt.code([1], "float32", [jt.ones(1)], cpu_header=header,
                        cpu_src="out0_p[0] = 1;").sync()
            self.assertIn("jt.code header:2:", str(caught.exception))

    def test_the_exit_code_is_the_compilers(self):
        """`pclose` returns a wait status: exit code 1 is 256, exit code 2 is 512.

        Reporting the raw status said "exit code 256", and anything above 256 --
        every nvcc failure, which exits with 2 -- was reported as the machine
        running out of memory, with the compiler's diagnostic dropped.
        """
        text = self._failure_text()
        self.assertNotIn("exit code 256", text)
        self.assertNotIn("out of memory", text)
        self.assertNotIn("overcommit", text)


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
            text.count("JT_TRACE_PY_VAR"), 1,
            "the same instruction is given more than once:\n" + text)

    def test_the_rerun_advice_does_not_suggest_a_rebuild(self):
        """`JT_SYNC` is a compile-time macro; `JT_SYNC_RUN` is the runtime flag."""
        text = self._failure_text()
        self.assertNotIn("JT_SYNC=", text)
        self.assertNotIn("JT_SYNC ", text)

    def test_no_rerun_advice_once_creation_sites_are_recorded(self):
        """With `trace_py_var` on, the report already names the creating line."""
        with jt.flag_scope(use_cuda=0, trace_py_var=3):
            source = jt.array(np.arange(5, dtype="float32"))
            failing = source[jt.array([99])]
            with self.assertRaises(RuntimeError) as caught:
                failing.sync()
        text = str(caught.exception)
        self.assertIn("test_error_message_quality.py", text,
                      "the creation site is missing from the report:\n" + text)
        self.assertNotIn("JT_TRACE_PY_VAR", text,
                         "advice to turn on what is already on:\n" + text)

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


class TestAFailedVarStaysFailed(unittest.TestCase):
    """The CPU getitem kernel clamps an out-of-range index, writes the clamped
    element and only then reports the fault. The output memory was therefore
    filled, and a second read took the "already on the host" shortcut and
    returned that element without a word."""

    def test_rereading_raises_again(self):
        with jt.flag_scope(use_cuda=0):
            source = jt.arange(3).float32() + 100
            source.sync()
            failing = source[jt.array([10])]
            for read in (failing.numpy, failing.numpy, failing.item):
                with self.subTest(read=read.__name__):
                    with self.assertRaises(RuntimeError) as caught:
                        read()
                    self.assertIn("out of bounds", str(caught.exception))


class TestShapeMessagesNameTheShapes(unittest.TestCase):
    """A shape error is only actionable if it says which shapes disagreed."""

    def _message(self, fn):
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(RuntimeError) as caught:
                fn()
        return str(caught.exception)

    def test_reshape(self):
        text = self._message(lambda: jt.ones((2, 3)).reshape(4, 2).sync())
        self.assertIn("[4,2,]", text)
        self.assertIn("[2,3,]", text)

    def test_integer_index(self):
        text = self._message(lambda: jt.ones((2, 3))[:, -5].sync())
        self.assertIn("-5", text)
        self.assertIn("dimension 1", text)
        self.assertNotIn("  ", text.splitlines()[0])

    def test_setitem(self):
        def assign():
            target = jt.zeros((2, 3))
            target[:] = jt.ones((2, 4))
            target.sync()
        text = self._message(assign)
        self.assertIn("[2,4,]", text)
        self.assertIn("[2,3,]", text)


class TestCreationRejectsAStringShape(unittest.TestCase):
    """Every str passes `is_dtype`, so `ones('abc')` became a dtype with an
    empty shape and failed later on `shape[0]` with an unrelated IndexError."""

    def test_a_lone_string(self):
        for fn in (jt.ones, jt.zeros, jt.empty):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(TypeError):
                    fn("abc")

    def test_a_string_dimension(self):
        with self.assertRaises(TypeError):
            jt.ones((2, "3"))

    def test_a_trailing_dtype_still_works(self):
        self.assertEqual(jt.ones(2, 3, "int32").dtype, "int32")
        self.assertEqual(list(jt.zeros((2, 3), "float64").shape), [2, 3])

    def test_an_unknown_dtype_names_the_function_called(self):
        """It used to be reported by the internal `unary()`, with its signature."""
        for fn in (jt.ones, jt.zeros, jt.empty):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(TypeError) as caught:
                    fn((2,), dtype="float99")
                text = str(caught.exception)
                self.assertTrue(text.startswith(fn.__name__ + "(): unknown dtype 'float99'"), text)
                self.assertNotIn("unary", text)


class TestCompileFailureNamesTheOperator(unittest.TestCase):
    """It began `parallel_compiler.cc:410: Error happened during compilation:`
    and named the operator as `Op(1:0:1:1:i0:o1:s0:g1,code->2)`."""

    def test_the_operator_is_named_and_the_wrapper_is_gone(self):
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(Exception) as caught:
                jt.code([1], "float32", [jt.ones(1)], cpu_src=BROKEN_KERNEL).sync()
        text = str(caught.exception)
        self.assertTrue(text.startswith("Compiling operator code"), text[:200])
        self.assertNotIn("Error happened during compilation", text)
        self.assertNotIn("Op(", text)


class TestUnknownDtypesAreNamed(unittest.TestCase):
    """Only `ones`/`zeros`/`empty` named an unknown dtype; `cast`, `array` and
    `randn` printed the signature listing of an internal binding."""

    def test_array_and_randn(self):
        for fn in (lambda: jt.array([1, 2], dtype="float99"),
                   lambda: jt.randn((2,), dtype="float99")):
            with self.assertRaises(TypeError) as caught:
                fn()
            text = str(caught.exception)
            self.assertTrue(text.startswith("unknown dtype 'float99'"), text)
            self.assertNotIn("checked_dtype_name", text)

    def test_cast_names_the_string_in_the_mismatch_report(self):
        with self.assertRaises(TypeError) as caught:
            jt.ones((2,)).cast("float99")
        first_lines = "\n".join(str(caught.exception).splitlines()[:2])
        self.assertIn("'float99' is not a dtype or operator name", first_lines)


class TestShapeErrorsNameTheCallersApi(unittest.TestCase):
    """Each of these used to surface from an internal step -- a tuple unpack,
    a reshape, a broadcasting multiply -- that the caller never wrote."""

    def _raises(self, fn, *needles):
        with self.assertRaises(jt.JittorShapeError) as caught:
            fn().sync()
        text = str(caught.exception).splitlines()[0]
        for needle in needles:
            self.assertIn(needle, text)
        return text

    def test_pooling_rank(self):
        self._raises(lambda: jt.nn.MaxPool2d(2)(jt.ones((8,))), "max_pool2d", "[8]")
        self._raises(lambda: jt.nn.MaxPool3d(2)(jt.ones((1, 4, 4, 4))), "max_pool3d")
        self._raises(lambda: jt.nn.MaxPool1d(2)(jt.ones((8,))), "pool1d", "[8]")

    def test_batch_norm_channels(self):
        self._raises(lambda: jt.nn.BatchNorm2d(3)(jt.ones((2, 4, 5, 5))),
                     "3 channels", "got 4")
        self._raises(lambda: jt.nn.batch_norm(jt.ones((2, 4, 3)), jt.zeros(3), jt.ones(3)),
                     "batch_norm", "got 4")

    def test_cross_entropy_batch(self):
        self._raises(lambda: jt.nn.cross_entropy_loss(jt.ones((4, 10)), jt.array([1, 2])),
                     "cross_entropy_loss", "4 rows", "2 elements")

    def test_concat_names_the_var_not_a_dim(self):
        text = self._raises(lambda: jt.concat([jt.ones((2, 3)), jt.ones((2, 2)), jt.ones((2, 4))], 0),
                            "var number 1")
        self.assertIsInstance(jt.JittorShapeError("x"), RuntimeError)

    def test_stack(self):
        self._raises(lambda: jt.stack([jt.ones((2, 3)), jt.ones((2, 4))]),
                     "equal size", "[2, 3]", "[2, 4]")

    def test_broadcast_has_no_internal_mask(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.ones((2, 3)).broadcast((4, 5)).sync()
        self.assertNotIn("bcast_mask", str(caught.exception))

    def test_a_reduce_dim_out_of_range_is_an_index_error(self):
        with self.assertRaises(IndexError):
            jt.ones((2, 3)).sum(5).sync()


class TestImportWarningsNameTheProgram(unittest.TestCase):
    """The unread-`JT_*` warning was reported at jittor's own `__init__.py`,
    the one place a reader cannot change."""

    def test_the_warning_points_at_the_import(self):
        import textwrap
        from _helpers.child_process import run_child_script
        result = run_child_script(textwrap.dedent('''
            import warnings
            warnings.simplefilter("always")
            import jittor
        '''), env={"JT_CACHE_NAME": "x"}, text=True, timeout=600,
            name="import_warning_location")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next((l for l in result.stderr.splitlines() if "JT_CACHE_NAME" in l), "")
        self.assertIn("import_warning_location", line, result.stderr)
        self.assertNotIn("jittor/__init__.py", line, result.stderr)


class TestLazyShimImportKeepsTheCause(unittest.TestCase):
    """`from jittor.compat.shim import activate` reported a failure inside the
    imported module as "cannot import name 'activate'", dropping the cause."""

    def test_an_attribute_error_inside_the_import_is_chained(self):
        import importlib
        from unittest import mock
        import jittor.compat.shim as shim
        broken = AttributeError("module 'jittor' has no attribute 'nn'")
        with mock.patch.object(importlib, "import_module", side_effect=broken):
            shim.__dict__.pop("activate", None)
            with self.assertRaises(ImportError) as caught:
                shim.__getattr__("activate")
        self.assertIn("has no attribute 'nn'", str(caught.exception))
        self.assertIs(caught.exception.__cause__, broken)


if __name__ == "__main__":
    unittest.main()
