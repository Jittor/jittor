# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``unique`` sorts by the value it was given, not by a truncated copy of it.

The CPU kernel's comparator read every element into an ``int``::

    int lhs = @input_flatten(a, i), rhs = @input_flatten(b, i);
    if (lhs != rhs) return lhs < rhs;

so 1.5 and 1.2 were "equal" and 2**32+7 and 7 were "equal". The pass that drops
duplicates afterwards compares the *real* values of neighbours, so it only ever
merges equal elements that the sort put next to each other. Truncation kept them
apart, and the result came back **containing duplicates and out of order** --
silently, and for the two most ordinary dtypes there are.

The comment above the CUDA dispatcher called the CPU path "the (correct) CPU
implementation" and routed float and wide-integer inputs to it, so on CUDA those
dtypes reached the truncating comparator *by design*.

numpy is the oracle: ``np.unique`` is an independent implementation of the same
contract (sorted uniques, plus inverse and counts).
"""

import unittest

import numpy as np

import jittor as jt


# Inputs chosen so that truncating the sort key to int collides. With the
# ordinary [1, 3, 2, 3] the two implementations agree and the defect is
# invisible, which is why it survived.
_COLLIDING = {
    "float32": np.array([1.5, 1.2, 1.5, -1.2, -1.5, 1.2], dtype="float32"),
    "float64": np.array([1.5, 1.2, 1.5, -1.2, -1.5, 1.2], dtype="float64"),
    "int64": np.array([2 ** 32 + 7, 7, 2 ** 32 + 7, 2 ** 33, 7],
                      dtype="int64"),
    # int32 fits an int, so this one always worked -- it is the control
    "int32": np.array([3, 1, 2, 1, 3], dtype="int32"),
}

_MULTI_COLUMN = {
    "float32": np.array([[1.5, 0.0], [1.2, 0.0], [1.5, 0.0]], dtype="float32"),
    "int64": np.array([[2 ** 32 + 7, 0], [7, 0], [2 ** 32 + 7, 0]],
                      dtype="int64"),
}


class _Unique:

    use_cuda = 0

    def _check(self, raw, dtype, dim=None):
        with jt.flag_scope(use_cuda=self.use_cuda):
            # jt.array() narrows float64/int64 numpy arrays unless told not to,
            # and narrowing would hide the very inputs under test
            x = jt.array(raw, dtype=dtype)
            self.assertEqual(str(x.dtype), dtype)
            values, inverse, counts = jt.unique(
                x, True, True, True, dim)
            got = (values.numpy(), inverse.numpy(), counts.numpy())
        want = np.unique(raw, return_inverse=True, return_counts=True, axis=dim)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1].reshape(want[1].shape), want[1])
        np.testing.assert_array_equal(got[2], want[2])

    def test_flat_values_that_collide_when_truncated(self):
        for dtype, raw in _COLLIDING.items():
            with self.subTest(dtype=dtype):
                self._check(raw, dtype)

    def test_along_a_dim_with_values_that_collide_when_truncated(self):
        for dtype, raw in _MULTI_COLUMN.items():
            with self.subTest(dtype=dtype):
                self._check(raw, dtype, dim=0)

    def test_result_is_sorted_and_free_of_duplicates(self):
        """The two properties the defect broke, stated on their own."""
        raw = _COLLIDING["float32"]
        with jt.flag_scope(use_cuda=self.use_cuda):
            got = jt.unique(jt.array(raw, dtype="float32")).numpy()
        self.assertEqual(len(got), len(set(got.tolist())),
                         "unique returned a duplicate")
        np.testing.assert_array_equal(got, np.sort(got))

    def test_uniform_dtype_coverage(self):
        """Every dtype answers the same question, including the narrow ones."""
        raw = np.array([3, 1, 2, 1, 3])
        for dtype in ("int8", "int16", "int32", "int64", "uint8",
                      "float16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=self.use_cuda):
                    got = jt.unique(jt.array(raw.astype(dtype),
                                             dtype=dtype)).numpy()
                np.testing.assert_array_equal(got, np.array([1, 2, 3], dtype))

    def test_empty_and_single(self):
        for raw in (np.array([], dtype="float32"),
                    np.array([7.5], dtype="float32")):
            with self.subTest(n=raw.size):
                with jt.flag_scope(use_cuda=self.use_cuda):
                    got = jt.unique(jt.array(raw, dtype="float32")).numpy()
                np.testing.assert_array_equal(got, np.unique(raw))


class TestUniqueCPU(_Unique, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestUniqueCUDA(_Unique, unittest.TestCase):
    use_cuda = 1


class TestUniqueHasOneImplementation(unittest.TestCase):

    def test_no_host_synchronisation_and_no_cpu_detour(self):
        """The dispatcher above the kernel is gone, and so is what it cost.

        It had four arms -- native int32 CUDA, "cast to int32 and recurse" for
        integers that fit, ``flag_scope(use_cuda=0)`` for everything else, and
        the CPU path -- and it chose between them with
        ``bool(((x <= 2147483647) & (x >= -2147483648)).all())``, a host
        synchronisation in the middle of a lazy graph, once per call.

        If this assertion ever has to be relaxed, say in the message which of
        the two it is: a value read back to the host, or a second
        implementation on another device.

        Parsed rather than grepped: a comment explaining the removed code would
        otherwise trip it.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(jt.misc.unique)))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        self.assertNotIn("bool", called, "a host synchronisation is back")
        keywords = {kw.arg for node in ast.walk(tree)
                    if isinstance(node, ast.Call) for kw in node.keywords}
        self.assertNotIn("use_cuda", keywords,
                         "unique switches devices behind the caller again")


if __name__ == "__main__":
    unittest.main()
