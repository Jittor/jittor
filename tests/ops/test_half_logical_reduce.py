# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``any`` and ``all`` of a floating-point Var, on the device.

The logical reductions ran on their input's dtype, and CUDA has no atomic OR
or AND for a floating-point type: the kernel did not compile. Transformers'
static KV cache asks ``any`` of a bfloat16 tensor, so a bf16 model could not
decode with it at all; ``any`` of a float32 tensor failed the same way.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestHalfLogicalReduce(unittest.TestCase):

    def test_any_and_all_of_half_precision(self):
        values = np.array([[0, 0, 0], [0, 2, 0], [1, 1, 1]], np.float32)
        with jt.flag_scope(use_cuda=1):
            for dtype in ("float16", "bfloat16", "float32", "float64"):
                x = jt.array(values).cast(dtype)
                self.assertTrue(bool(x.any().numpy()))
                self.assertFalse(bool(x.all().numpy()))
                np.testing.assert_array_equal(x.any(1).numpy(), [False, True, True])
                np.testing.assert_array_equal(x.all(1).numpy(), [False, False, True])
                nan = jt.array(np.array([np.nan, 0], np.float32)).cast(dtype)
                self.assertTrue(bool(nan.any().numpy()))    # NaN is true, as in torch


if __name__ == "__main__":
    unittest.main()
