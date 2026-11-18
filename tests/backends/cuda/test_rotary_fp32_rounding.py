"""Check the rounding of both halves of the standalone FP32 NeoX rotation."""
import unittest

import jittor as jt
import numpy as np
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA required')
class TestRotaryRounding(unittest.TestCase):
    def test_query_and_key_match_single_rounding(self):
        rng = np.random.RandomState(39)
        q, k, table = [rng.normal(size=s).astype(np.float32)
                       for s in ((3, 16), (3, 8), (3, 8))]
        c, s = np.split(table[:, None, :], 2, axis=-1)
        with jt.flag_scope(use_cuda=1, cuda_kernel_math='strict'), jt.no_grad():
            actual = jt.nn.rotary_embedding(jt.array(np.arange(3, dtype=np.int64)),
                                             jt.array(q), jt.array(k), jt.array(table),
                                             head_size=8, rotary_dim=8, is_neox=True)
            for observed, raw in zip(actual, (q, k)):
                x, y = np.split(raw.reshape(3, -1, 8), 2, axis=-1)
                # Binary64 represents each FP32 product exactly. Round the
                # standalone product, then round the fused multiply-add once.
                left = x.astype(np.float64) * c - (y * s)
                right = x.astype(np.float64) * s + (y * c)
                expected = np.concatenate((left, right), axis=-1).astype(np.float32)
                np.testing.assert_array_equal(observed.numpy(), expected.reshape(raw.shape))
