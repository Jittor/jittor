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

import os
import subprocess
import sys
import textwrap
import unittest

import jittor as jt
import jittor_core
import numpy as np


def run_in_subprocess(body):
    """Run ``body`` in a fresh interpreter and return the CompletedProcess.

    Used for cases whose pre-fix behaviour is a segfault: a crash inside the
    test process would take the whole session down with it, so the assertion
    has to be made on a child process' return code.

    The child is pinned to *this* process' jittor package.  pytest puts the
    checkout on ``sys.path`` via ``pythonpath`` in pyproject.toml, which a bare
    subprocess does not inherit -- without this it would silently import
    whatever jittor is installed in site-packages and test the wrong tree.
    """
    source = "import jittor as jt\nimport jittor_core\n" + textwrap.dedent(body)
    env = dict(os.environ)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(jt.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=1800,
    )


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


if __name__ == "__main__":
    unittest.main()
