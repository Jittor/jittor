# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""CPython-protocol conformance of the generated pyjt bindings.

These are not tests of any operator: they cover the contract between the
generated ``jittor_core`` types and CPython itself -- object lifetime, the
``tp_new``/``tp_init``/``tp_dealloc`` sequence, argument conversion errors.
A break here shows up as a crash or as silently wrong data, never as a normal
Python exception, which is why each case is spelled out separately.
"""

import gc
import signal
import textwrap
import unittest

import jittor as jt
import jittor_core
import numpy as np

from _helpers.child_process import run_python_child


def run_in_subprocess(body):
    """Run ``body`` in a fresh interpreter and return the CompletedProcess.

    Used for cases whose pre-fix behaviour is a segfault: a crash inside the
    test process would take the whole session down with it, so the assertion
    has to be made on a child process' return code.

    ``crash_isolated`` is what makes that true. Jittor's process-level SIGCHLD
    handler quick-exits the parent when a direct child dies from a signal, so
    without the shell in between a regressed case here would not fail -- it
    would make pytest disappear mid-run with no output (6.C31).
    """
    source = "import jittor as jt\nimport jittor_core\n" + textwrap.dedent(body)
    return run_python_child(["-c", source], text=False, merge_stderr=True,
                            timeout=1800, crash_isolated=True)


class TestConstructionFailureDealloc(unittest.TestCase):
    """A failed ``tp_init`` must not leave a C++ destructor to run on zeros."""

    def test_ring_buffer_no_args_raises_instead_of_crashing(self):
        # ``PyType_GenericNew`` zeroes the instance, then ``tp_init`` finds no
        # matching overload and returns -1, and CPython goes straight to
        # ``tp_dealloc``.  Without a "constructed" flag the generated dealloc
        # runs ``~PyMultiprocessRingBuffer()`` on that zeroed storage, which
        # dereferences a null ``rb`` -> segfault.
        proc = run_in_subprocess(
            """
            try:
                jittor_core.RingBuffer()
            except Exception:
                print("RAISED")
            else:
                print("NO_ERROR")
            print("SURVIVED")
            """
        )
        output = proc.stdout.decode("utf8", "replace")
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("SURVIVED", output)
        self.assertIn("RAISED", output)

    def test_ring_buffer_bad_argument_types_raise_instead_of_crashing(self):
        proc = run_in_subprocess(
            """
            for args in [("not an int",), (1, 2, 3, 4)]:
                try:
                    jittor_core.RingBuffer(*args)
                except Exception:
                    pass
            print("SURVIVED")
            """
        )
        output = proc.stdout.decode("utf8", "replace")
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("SURVIVED", output)

    def test_ring_buffer_still_works_when_constructed(self):
        # The guard must not break the normal path.
        rb = jittor_core.RingBuffer(1024)
        rb.push(42)
        assert rb.pop() == 42
        del rb

    def test_var_lifetime_unaffected(self):
        # VarHolder carries the same flag word; ordinary create/destroy must
        # still work and still free.
        for _ in range(100):
            v = jt.array([1.0, 2.0])
            v.foo = 1
            assert v.foo == 1
            del v


class TestSliceUnpack(unittest.TestCase):
    """``PySlice_Unpack`` failures must not become slice bounds.

    CPython returns -1 and leaves ``start``/``stop``/``step`` untouched when a
    slice is unusable (``step == 0``, or an ``__index__`` that raises), so the
    converter has to check the return value before reading them -- otherwise
    uninitialised stack is what reaches getitem/setitem.
    """

    def setUp(self):
        self.a = jt.array(np.arange(20, dtype="float32").reshape(4, 5))

    def test_zero_step_getitem_raises_value_error(self):
        for index in (
            lambda a: a[::0],
            lambda a: a[1:3:0],
            lambda a: a[0, ::0],
            lambda a: a[::0, 0],
            lambda a: a[::0, ::0],
        ):
            with self.assertRaises(ValueError):
                index(self.a)

    def test_zero_step_setitem_raises_value_error(self):
        b = jt.array(np.arange(20, dtype="float32").reshape(4, 5))
        with self.assertRaises(ValueError):
            b[::0] = 1.0
        # the failed store must not have touched the data
        np.testing.assert_array_equal(
            b.numpy(), np.arange(20, dtype="float32").reshape(4, 5))

    def test_failing_index_protocol_propagates(self):
        class Boom:
            def __index__(self):
                raise RuntimeError("boom")

        with self.assertRaises(Exception) as ctx:
            self.a[:: Boom()]
        self.assertNotIsInstance(ctx.exception, SystemError)

    def test_normal_slices_still_work(self):
        np_a = np.arange(20, dtype="float32").reshape(4, 5)
        np.testing.assert_array_equal(self.a[::2].numpy(), np_a[::2])
        np.testing.assert_array_equal(self.a[::-1].numpy(), np_a[::-1])
        np.testing.assert_array_equal(self.a[1:3, ::2].numpy(), np_a[1:3, ::2])


class TestInstanceDictParticipatesInGC(unittest.TestCase):
    """A type with an instance ``__dict__`` must be collectable.

    ``tp_dictoffset`` was set without ``Py_TPFLAGS_HAVE_GC``/traverse/clear, so
    any cycle closed through the dict -- ``v.foo = v``, or the ``grad``/``_base``
    back-pointers a torch shim hangs off a tensor -- was never collected, and
    each leaked wrapper pinned its whole graph and the device memory behind it.
    """

    def test_var_type_is_gc_enabled(self):
        Py_TPFLAGS_HAVE_GC = 1 << 14
        self.assertTrue(jt.Var.__flags__ & Py_TPFLAGS_HAVE_GC)
        self.assertTrue(gc.is_tracked(jt.array([1.0])))

    def test_self_referencing_var_is_collected(self):
        gc.collect()
        jt.gc()
        before = jt.liveness_info()["lived_vars"]
        for _ in range(50):
            v = jt.array([1.0, 2.0])
            v.self_ref = v          # cycle through the instance dict
            del v
        gc.collect()
        jt.gc()
        after = jt.liveness_info()["lived_vars"]
        self.assertEqual(after, before)

    def test_two_var_cycle_is_collected(self):
        gc.collect()
        jt.gc()
        before = jt.liveness_info()["lived_vars"]
        for _ in range(20):
            a = jt.array([1.0])
            b = jt.array([2.0])
            a.peer = b
            b.peer = a
            del a, b
        gc.collect()
        jt.gc()
        self.assertEqual(jt.liveness_info()["lived_vars"], before)

    def test_attributes_and_dict_still_work(self):
        v = jt.array([1.0])
        v.foo = 42
        self.assertEqual(v.foo, 42)
        self.assertEqual(v.__dict__["foo"], 42)
        self.assertEqual(vars(v)["foo"], 42)
        del v.foo
        self.assertFalse(hasattr(v, "foo"))

class TestScalarConversionBuffer(unittest.TestCase):
    """A converted Python scalar must survive until its consumer copies it.

    Scalar -> 1-element-array conversion used to hand back the address of one
    process-wide union, so whatever ran between the conversion and the copy
    could overwrite the value.  ``Var.data = 2.0`` is exactly that shape: the
    binding converts 2.0, then ``set_data`` syncs the graph, and a python
    callback executing inside that sync (numpy_code, fetch) converts scalars of
    its own.
    """

    def test_set_data_scalar_survives_callback_during_sync(self):
        def fwd(np_out, data):
            # runs during the sync that set_data performs
            jt.array(7.0)
            data["outputs"][0][:] = 1.0

        x = jt.numpy_code([(1,)], ["float32"], [jt.zeros(1)], fwd)[0]
        b = x + 0.0
        b.data = 2.0
        np.testing.assert_allclose(b.numpy(), [2.0])

    def test_scalar_arrays_are_correct(self):
        np.testing.assert_allclose(jt.array(1.5).numpy(), 1.5)
        np.testing.assert_array_equal(jt.array(7).numpy(), 7)
        np.testing.assert_array_equal(jt.array(True).numpy(), True)
        self.assertEqual(str(jt.array(1.5).dtype), "float32")
        self.assertEqual(str(jt.array(7).dtype), "int32")
        self.assertEqual(str(jt.array(True).dtype), "bool")

class TestKeywordArguments(unittest.TestCase):
    """Keyword arguments must be mapped to parameter slots before type checks.

    Three failures came out of doing it the other way round: a signature with
    no keyword-fillable parameter never looked at ``kw`` at all and dropped
    whatever was passed; overload selection probed ``args[tid]``, which under
    FASTCALL holds a *keyword value* once ``tid >= n``, so the answer depended
    on the order the caller wrote its keywords; and the single
    ``PyErr_Occurred()`` check ran before the keyword conversions, so an
    overflowing keyword value was used anyway.
    """

    def setUp(self):
        self.x = jt.array(np.arange(12, dtype="float32").reshape(3, 4))

    def test_unknown_keyword_is_rejected(self):
        # detach() takes no keyword-fillable parameter at all; the keyword used
        # to be dropped and the call to succeed with default semantics.
        with self.assertRaises(Exception):
            self.x.detach(non_blocking=True)

    def test_keyword_order_does_not_change_the_overload(self):
        a = self.x.sum(dim=1, keepdims=True)
        b = self.x.sum(keepdims=True, dim=1)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(a.numpy(), b.numpy())
        np.testing.assert_allclose(
            a.numpy(), self.x.numpy().sum(axis=1, keepdims=True))

    def test_overflowing_keyword_value_raises(self):
        # PyLong_AsLong overflows, sets OverflowError and returns -1; the old
        # code ignored it and reduced over dim -1 instead.
        with self.assertRaises(Exception) as ctx:
            self.x.sum(dim=2 ** 40)
        self.assertNotIsInstance(ctx.exception, SystemError)

    def test_overflowing_positional_value_raises(self):
        with self.assertRaises(Exception):
            self.x.sum(2 ** 40)

    def test_keywords_that_do_exist_still_work(self):
        np.testing.assert_allclose(
            self.x.sum(dim=0).numpy(), self.x.numpy().sum(axis=0))
        np.testing.assert_allclose(
            self.x.sum(dim=0, keepdims=True).numpy(),
            self.x.numpy().sum(axis=0, keepdims=True))
        # `keepdim` is accepted as an alias of `keepdims`
        np.testing.assert_allclose(
            self.x.sum(dim=0, keepdim=True).numpy(),
            self.x.numpy().sum(axis=0, keepdims=True))

    def test_duplicate_value_for_one_parameter_is_rejected(self):
        with self.assertRaises(Exception):
            self.x.sum(0, dim=1)


class TestNonStdExceptionAtTheBoundary(unittest.TestCase):
    """A C++ exception outside the std hierarchy must not terminate CPython."""

    def test_custom_op_throwing_a_non_std_exception(self):
        # The generated bindings are extern "C" functions: an exception that
        # reaches that boundary calls std::terminate and the interpreter dies
        # with no traceback, so this has to run in a child.  ``run_in_subprocess``
        # asks for crash isolation, without which the child's abort trips
        # jittor's SIGCHLD handler and takes this session down too.
        proc = run_in_subprocess("""
            jt.flags.use_cuda = 0
            try:
                x = jt.code([1], "float32", [], cpu_src='throw 42;')
                x.sync()
                print("NO-RAISE")
            except Exception:
                print("RAISED")
            print("SURVIVED")
        """)
        output = proc.stdout.decode("utf8", "replace")
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("RAISED", output)
        self.assertIn("SURVIVED", output)

if __name__ == "__main__":
    unittest.main()


class TestCrashIsolationSurvivesTheSigchldHandler(unittest.TestCase):
    """A child that dies by signal must fail the test, not delete the session.

    Jittor installs a process-level ``SIGCHLD`` handler that quick-exits the
    parent when a *direct* child dies from a signal. Every crash test in this
    file depends on the opposite: that a child can abort and be asserted on.
    Without ``crash_isolated`` the handler fires inside pytest and pytest
    vanishes mid-run with no output -- which reads as a broken runner rather
    than a failing test, and is what 6.C31 records.

    This case is the guard on that guard: it makes the child abort on purpose.
    If the isolation regresses, the whole session disappears here, in a test
    whose name says why.
    """

    def test_an_aborting_child_reports_its_signal_and_leaves_pytest_alive(self):
        proc = run_python_child(
            ["-c", "import os, signal; os.kill(os.getpid(), signal.SIGABRT)"],
            text=False, merge_stderr=True, timeout=300, crash_isolated=True)
        # The shell exits 128 + signo, so the crash is still assertable.
        self.assertEqual(proc.returncode, 128 + int(signal.SIGABRT))
        # Reaching this line at all is the other half of the assertion.
        self.assertTrue(True)
