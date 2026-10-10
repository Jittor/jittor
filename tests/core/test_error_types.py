# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Errors from the C++ core reach Python with a class that says what they are.

The core already told a bad index (UserError) from a broken invariant
(InternalInvariantError), but the executor rethrew every failure as a plain
std::runtime_error and the bindings turned every C++ exception into
RuntimeError. ``except IndexError`` could not catch an out-of-range index and
nothing could tell "my arguments are wrong" from "Jittor is broken".

Every class is still a RuntimeError, so code written against the old behaviour
keeps catching what it caught.
"""

import pickle
import unittest

import numpy as np

import jittor as jt


class TestErrorClasses(unittest.TestCase):

    def _raised(self, fn):
        with jt.flag_scope(use_cuda=0):
            with self.assertRaises(Exception) as caught:
                fn()
        return caught.exception

    def test_an_integer_index_out_of_range_is_an_index_error(self):
        error = self._raised(lambda: jt.ones((2, 3))[5])
        self.assertIsInstance(error, jt.JittorIndexError)
        self.assertIsInstance(error, IndexError)
        self.assertIsInstance(error, RuntimeError)

    def test_too_many_indices_is_an_index_error(self):
        self.assertIsInstance(self._raised(lambda: jt.ones((2, 3))[0, 1, 2]),
                              IndexError)

    def test_the_kind_survives_the_executor(self):
        """A Var index is only checked when the kernel runs, and the executor
        wraps the failure with the op it came from before rethrowing it."""
        source = jt.array(np.arange(3, dtype="float32"))
        error = self._raised(lambda: source[jt.array([10])].numpy())
        self.assertIsInstance(error, jt.JittorIndexError)
        self.assertIn("op: getitem", str(error))

    def test_an_executor_failure_triggered_by_argument_conversion(self):
        """Reported as the operator's failure, not as bad arguments.

        Assigning a Var to `.data` reads it while converting the argument,
        before any overload has matched. The binding used to tell the two
        apart by looking for "exec_runner.cc:" in the text.
        """
        source = jt.array(np.arange(3, dtype="int32"))
        target = jt.zeros(1, dtype="int32")

        def assign():
            target.data = source[jt.array([10])]

        error = self._raised(assign)
        self.assertIsInstance(error, jt.JittorIndexError)
        self.assertNotIn("match none of its signatures", str(error))

    def test_setitem_with_an_index_out_of_range(self):
        def assign():
            target = jt.zeros(3)
            target[jt.array([10])] = 1.0
            target.sync()
        self.assertIsInstance(self._raised(assign), IndexError)

    def test_a_shape_mismatch_is_a_plain_jittor_error(self):
        error = self._raised(lambda: (jt.ones((3, 4)) + jt.ones((5, 6))).sync())
        self.assertIsInstance(error, jt.JittorError)
        self.assertNotIsInstance(error, (IndexError, TypeError))
        self.assertNotIsInstance(error, jt.JittorInternalError)

    def test_no_matching_overload_is_a_type_error(self):
        error = self._raised(lambda: setattr(jt.flags, "use_cuda", "yes"))
        self.assertIsInstance(error, jt.JittorTypeError)
        self.assertIsInstance(error, TypeError)
        self.assertIsInstance(error, RuntimeError)

    def test_the_classes_pickle(self):
        """A worker process hands its exception to the parent by pickling."""
        error = self._raised(lambda: jt.ones((2, 3))[5])
        self.assertIs(type(pickle.loads(pickle.dumps(error))), jt.JittorIndexError)

    def test_the_hierarchy(self):
        for cls in (jt.JittorInternalError, jt.JittorIndexError, jt.JittorTypeError,
                    jt.JittorShapeError):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, jt.JittorError))
        self.assertTrue(issubclass(jt.JittorError, RuntimeError))

    def test_matmul_and_conv_raise_the_same_shape_error(self):
        """matmul raised RuntimeError and conv ValueError for the same kind of
        mistake. One class that is both keeps either `except` working."""
        cases = {
            "matmul": lambda: jt.matmul(jt.ones((2, 3)), jt.ones((5, 4))),
            "linear": lambda: jt.nn.Linear(4, 5)(jt.ones((2, 3))),
            "conv2d": lambda: jt.nn.Conv2d(3, 8, 3)(jt.ones((1, 4, 8, 8))),
            "conv2d rank": lambda: jt.nn.Conv2d(3, 8, 3)(jt.ones((4, 8, 8))),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                error = self._raised(fn)
                self.assertIs(type(error), jt.JittorShapeError)
                self.assertIsInstance(error, RuntimeError)
                self.assertIsInstance(error, ValueError)

    def test_shapes_in_conv_messages_are_lists(self):
        error = self._raised(lambda: jt.nn.Conv2d(3, 8, 3)(jt.ones((1, 4, 8, 8))))
        self.assertIn("input shape [1, 4, 8, 8]", str(error))


if __name__ == "__main__":
    unittest.main()
