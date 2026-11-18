"""Regress the gather and small-sort tie order on CUDA inference rows."""
import unittest

import jittor as jt
import numpy as np
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'CUDA required')
class TestWideTopK(unittest.TestCase):
    def test_cutoff_ties_and_sorted_winner_order(self):
        values = np.zeros((1, 20000), dtype=np.float32)
        values[0, [10, 20, 30, 40, 50, 60]] = [7, 5, 7, 7, 3, 3]
        with jt.flag_scope(use_cuda=1), jt.no_grad():
            scores, ids = jt.topk(jt.array(values), 5, dim=-1)
            np.testing.assert_array_equal(ids.numpy(), [[40, 30, 10, 20, 50]])
            np.testing.assert_array_equal(scores.numpy(), [[7, 7, 7, 5, 3]])
