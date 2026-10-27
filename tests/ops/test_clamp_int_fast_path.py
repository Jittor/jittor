from _helpers import capability as _test_capability
import unittest
from unittest import mock

import jittor as jt
import numpy as np
from jittor._core import var as var_module


_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
# Bounds that sit inside, on, and past the range of the narrow dtypes: the
# general path casts each bound to x's dtype first, so 300 wraps for int8.
_BOUNDS = ((0, 255), (-3, 3), (5, 5), (0, 300), (-200, -100))


def _observe(dtype, bounds):
    values = np.arange(-400, 400, 7)
    x = jt.array(values).cast(dtype)
    result = x.clamp(*bounds)
    return result.numpy(), tuple(result.shape), str(result.dtype)


class TestClampIntFastPath(unittest.TestCase):
    """An integer tensor clamped by two ordered Python ints must give exactly
    what the general clamp gives.

    `_clamp_cpu` answers that call with `maximum`/`minimum` over bounds cast to
    the tensor's dtype, on the claim that for an integer the general path's
    compare-and-select pair computes the same thing. Each case is evaluated with
    the fast path and again with it switched off, and the values, shape and
    dtype must agree.
    """

    def _check(self):
        for dtype in _DTYPES:
            for bounds in _BOUNDS:
                with self.subTest(dtype=dtype, bounds=bounds):
                    fast = _observe(dtype, bounds)
                    with mock.patch.object(var_module, "_CLAMP_INT_DTYPES", frozenset()):
                        full = _observe(dtype, bounds)
                    np.testing.assert_array_equal(fast[0], full[0])
                    self.assertEqual(fast[1:], full[1:])

    def test_matches_general_path_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self._check()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
    def test_matches_general_path_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self._check()

    def test_declined_calls_keep_their_meaning(self):
        # Reversed bounds give Torch's all-max answer; a float tensor still
        # carries NaN through; a bool tensor keeps its dtype.
        x = jt.array(np.array([-2, 0, 7], np.int32))
        np.testing.assert_array_equal(x.clamp(5, 1).numpy(), [1, 1, 1])
        f = jt.array(np.array([np.nan, -1.0, 9.0], np.float32)).clamp(0, 5).numpy()
        self.assertTrue(np.isnan(f[0]))
        np.testing.assert_array_equal(f[1:], [0.0, 5.0])
        self.assertEqual(str(jt.array(np.array([True, False])).clamp(0, 1).dtype), "bool")


if __name__ == "__main__":
    unittest.main()
