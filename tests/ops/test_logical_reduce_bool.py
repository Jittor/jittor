# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``any`` and ``all`` of a bool Var reduce straight into a bool.

``any_``/``all_`` reduce a bool input into int32, and ``any``/``all`` cast
that back to bool: a second kernel behind every ``if mask.any():``. On CPU
and CUDA the reduction now keeps the bool, and CUDA updates the one output
every thread meets at with a 32-bit atomic on the word holding it.
"""

import unittest

import numpy as np

import jittor as jt
from _helpers.common import JittorTestCase, kernel_launches
from _helpers.device_types import instantiate_device_type_tests


class TestLogicalReduceBool(JittorTestCase):

    def test_matches_numpy(self, device):
        rng = np.random.RandomState(0)
        for shape in ((), (0,), (3, 0), (1,), (5, 1, 4), (2, 3, 4, 5)):
            a = rng.rand(*shape) < 0.3 if shape else np.array(True)
            x = jt.array(a)
            nd = len(shape)
            dims = [()] + list(range(-nd, nd))
            if nd >= 2:
                dims += [(0, -1), [0], (nd - 1,)]
            for dim in dims:
                axis = None if dim == () else (tuple(dim) if isinstance(dim, (list, tuple)) else dim)
                for keepdims in (False, True):
                    for name in ("any", "all"):
                        got = getattr(x, name)(dim, keepdims=keepdims)
                        want = getattr(np, name)(a, axis=axis, keepdims=keepdims)
                        self.assertEqual(got.dtype, "bool")
                        np.testing.assert_array_equal(got.numpy().reshape(np.shape(want)), want)

    def test_every_thread_meets_at_one_output(self, device):
        # Enough elements for every block of the reduction, and the bytes of
        # a bool output that share one 32-bit word with its neighbours.
        rng = np.random.RandomState(1)
        for n in (4096, 1 << 20):
            for p in (0.0, 1e-6, 0.5, 1.0):
                a = rng.rand(n) < p
                if p == 1e-6:
                    a[:] = False
                    a[n // 3] = True
                x = jt.array(a)
                self.assertEqual(bool(x.any()), bool(a.any()))
                self.assertEqual(bool(x.all()), bool(a.all()))
        a = rng.rand(7, 4096) < 0.0005
        np.testing.assert_array_equal(jt.array(a).any(1).numpy(), a.any(1))
        np.testing.assert_array_equal(jt.array(~a).all(1).numpy(), (~a).all(1))

    def test_a_truth_test_is_one_reduction(self, device):
        m = jt.array(np.arange(240) % 7 == 0)
        m.sync()
        count, value = kernel_launches(lambda: bool(m.any()))
        self.assertTrue(value)
        self.assertEqual(count, 1)
        count, value = kernel_launches(lambda: bool(m.all()))
        self.assertFalse(value)
        self.assertEqual(count, 1)

    def test_the_result_is_placed_as_the_native_op_places_it(self, device):
        # A Var placed on the CPU explicitly, whatever the runtime device.
        token = jt.core._set_construction_placement(0, 0)
        try:
            x = jt.array(np.array([[False, True, False], [False, False, False]]))
        finally:
            jt.core._restore_construction_placement(token)
        self.assertEqual(x.placement_backend, 0)
        for keepdims in (False, True):
            got = x.any(1, keepdims=keepdims)
            want = jt.ops.any_(x, 1, keepdims=keepdims).bool()
            jt.sync([got, want])
            self.assertEqual(got.placement_backend, want.placement_backend)
            self.assertEqual(got.location(), want.location())
            np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_other_dtypes_and_errors_are_unchanged(self, device):
        f = jt.array(np.array([0.0, np.nan, 0.0], np.float32))
        self.assertTrue(bool(f.any()))       # NaN is true, as in torch
        self.assertEqual(f.any().dtype, "bool")
        i = jt.array(np.array([1, 2, 0], np.int32))
        self.assertFalse(bool(i.all()))
        self.assertEqual(i.all().dtype, "bool")
        # The native ops keep their int32 result.
        self.assertEqual(jt.ops.any_(jt.array(np.ones(3, bool))).dtype, "int32")
        x = jt.array(np.ones((2, 3), bool))
        for bad in (5, (0, 7)):
            with self.assertRaisesRegex(Exception, "Reduce dim out of range"):
                x.any(bad).sync()


instantiate_device_type_tests(TestLogicalReduceBool, globals())


if __name__ == "__main__":
    unittest.main()
