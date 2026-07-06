# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# Maintainers:
#     Jittor contributors.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestStackFast(unittest.TestCase):

    def test_no_grad_cuda_stack_matches_numpy(self):
        rng = np.random.RandomState(241)

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            for n in (2, 3):
                for dim in (0, 1, -1, 3):
                    arrays = [
                        rng.randn(2, 3, 5).astype("float16")
                        for _ in range(n)
                    ]
                    out = jt.stack([jt.array(a) for a in arrays], dim=dim)
                    ref = np.stack(arrays, axis=dim)
                    np.testing.assert_array_equal(out.numpy(), ref)

    def test_no_grad_cuda_unbind_matches_numpy(self):
        rng = np.random.RandomState(242)

        with jt.flag_scope(use_cuda=1), jt.no_grad():
            for shape, dim in (
                ((8, 3, 12, 128), 1),
                ((1, 16, 3, 12, 128), 2),
                ((8, 2, 12, 128), -3),
            ):
                arr = rng.randn(*shape).astype("float16")
                outs = jt.unbind(jt.array(arr), dim=dim)
                refs = np.split(arr, arr.shape[dim], axis=dim)
                refs = [np.squeeze(ref, axis=dim) for ref in refs]
                self.assertEqual(len(outs), len(refs))
                for out, ref in zip(outs, refs):
                    np.testing.assert_array_equal(out.numpy(), ref)


if __name__ == "__main__":
    unittest.main()
