# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""An odd-length draw must not write (or consume) one element too many.

``curand_random`` used to round the element count up to an even number for
both distributions, writing one element past the end of the output and relying
on the allocator to have left slack there.  The extra element is also drawn
from the generator, so it is observable without a memory checker: two uniform
draws of odd length stop continuing the stream that a single draw of the
combined length produces.
"""
import unittest

import numpy as np

import jittor as jt


def _draw(sizes, dtype="float32", type="uniform"):
    jt.set_seed(0)
    return np.concatenate([
        jt.random((size,), dtype, type).numpy().astype("float64")
        for size in sizes
    ])


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCurandOddLength(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._use_cuda

    def test_odd_uniform_draw_consumes_exactly_its_length(self):
        for split, total in (([3, 4], 7), ([1, 2], 3), ([5, 6], 11)):
            np.testing.assert_allclose(_draw(split), _draw([total]),
                                       err_msg="split %s vs %d" % (split, total))

    def test_even_uniform_draw_is_unaffected(self):
        np.testing.assert_allclose(_draw([4, 4]), _draw([8]))

    def test_odd_uniform_draw_float64(self):
        np.testing.assert_allclose(_draw([3, 4], "float64"),
                                   _draw([7], "float64"))

    def test_odd_normal_draw_is_well_formed(self):
        # The last element comes from a scratch buffer now; it must be a real
        # sample, not whatever was in the tail of the allocation.
        values = jt.random((4097,), "float32", "normal").numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(abs(values.mean()), 0.1)
        self.assertLess(abs(values.std() - 1.0), 0.1)
        tails = [float(jt.random((5,), "float32", "normal").numpy()[-1])
                 for _ in range(8)]
        self.assertGreater(len(set(tails)), 1)

    def test_unsupported_dtype_is_rejected_clearly(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.random((5,), "int32").sync()
        self.assertIn("float32", str(caught.exception))

    def test_low_precision_still_goes_through_a_float32_draw(self):
        for dtype in ("float16", "bfloat16"):
            values = jt.random((5,), dtype).numpy().astype("float64")
            self.assertEqual(values.shape, (5,))
            self.assertTrue(((values >= 0) & (values <= 1)).all())


if __name__ == "__main__":
    unittest.main()
