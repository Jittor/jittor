# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error

class TestFlags(unittest.TestCase):
    def test_error(self):
        def check(): jt.flags.asdasd=1
        expect_error(
            check,
            exc_type=AttributeError,
            match=r"no attribute 'asdasd'",
        )
    
    def test_get_set(self):
        prev = jt.flags.log_v
        jt.flags.log_v=1
        assert jt.flags.log_v == 1
        jt.flags.log_v=prev
        assert jt.flags.log_v == prev
    
    def test_scope(self):
        prev = jt.flags.log_v
        with jt.flag_scope(log_v=1):
            assert jt.flags.log_v == 1
        assert jt.flags.log_v == prev


class TestFlagScopeIsReentrant(unittest.TestCase):
    """A flag_scope kept its saved values in ONE attribute on the instance.

    Entering the same scope object again before leaving it overwrote the outer
    entry's backup with the inner one's, and the outer ``__exit__`` then
    restored the *inner* scope's values -- permanently, for the rest of the
    process.

    ``_call_no_record_scope.__call__`` closes over a single scope instance, so
    a **recursive** function decorated with ``@jt.no_grad()`` walks straight
    into it. The result is ``jt.flags.no_grad`` stuck at 1: every later
    ``jt.grad`` returns zeros, nothing raises, and training simply stops
    learning.
    """

    def setUp(self):
        self._saved = jt.flags.no_grad

    def tearDown(self):
        jt.flags.no_grad = self._saved

    def test_recursive_no_grad_decorator_does_not_leak(self):
        jt.flags.no_grad = 0

        @jt.no_grad()
        def recurse(n):
            assert jt.flags.no_grad, "inside the scope no_grad must be on"
            if n > 0:
                return recurse(n - 1)
            return n

        recurse(3)
        self.assertFalse(
            jt.flags.no_grad,
            "no_grad leaked out of a recursive @jt.no_grad() function")

    def test_the_leak_silently_kills_autodiff(self):
        # the consequence, asserted directly: this is what the user sees
        jt.flags.no_grad = 0

        @jt.no_grad()
        def recurse(n):
            return recurse(n - 1) if n > 0 else n

        recurse(1)
        x = jt.array(np.array([3.0], dtype="float32"))
        x.start_grad()
        g = jt.grad((x * x).sum(), [x])[0]
        np.testing.assert_allclose(
            g.numpy(), [6.0], rtol=1e-6,
            err_msg="autodiff silently returned zeros because no_grad leaked")

    def test_same_scope_instance_nested(self):
        jt.flags.no_grad = 0
        scope = jt.flag_scope(no_grad=1)
        with scope:
            self.assertTrue(jt.flags.no_grad)
            with scope:
                self.assertTrue(jt.flags.no_grad)
            self.assertTrue(jt.flags.no_grad,
                            "leaving the inner entry must not leave the outer")
        self.assertFalse(jt.flags.no_grad)

    def test_deeply_nested_restores_each_level(self):
        prev = jt.flags.log_v
        try:
            jt.flags.log_v = 0
            s1 = jt.flag_scope(log_v=1)
            s2 = jt.flag_scope(log_v=2)
            with s1:
                self.assertEqual(jt.flags.log_v, 1)
                with s2:
                    self.assertEqual(jt.flags.log_v, 2)
                    with s1:
                        self.assertEqual(jt.flags.log_v, 1)
                    self.assertEqual(jt.flags.log_v, 2)
                self.assertEqual(jt.flags.log_v, 1)
            self.assertEqual(jt.flags.log_v, 0)
        finally:
            jt.flags.log_v = prev

    def test_enable_grad_decorator_is_reentrant_too(self):
        jt.flags.no_grad = 1

        @jt.enable_grad()
        def recurse(n):
            assert not jt.flags.no_grad
            return recurse(n - 1) if n > 0 else n

        recurse(2)
        self.assertTrue(jt.flags.no_grad,
                        "enable_grad must restore the no_grad it found")

    def test_decorator_preserves_the_function_identity(self):
        @jt.no_grad()
        def documented(a, b=2):
            """docstring kept?"""
            return a + b

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "docstring kept?")
        self.assertEqual(documented(1), 3)
        self.assertEqual(documented(1, b=5), 6)

    def test_exception_inside_the_scope_still_restores(self):
        jt.flags.no_grad = 0

        @jt.no_grad()
        def boom(n):
            if n > 0:
                return boom(n - 1)
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            boom(2)
        self.assertFalse(jt.flags.no_grad,
                         "an exception must not strand the flag")


if __name__ == "__main__":
    unittest.main()
