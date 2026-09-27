"""A half-precision convolution's cached filter stays out of copies of the weight.

Without a backward, the cuDNN path keeps an OHWI copy of a half-precision
filter on the weight (``jittor/nn/backends/cudnn.py``). It is derived and as
large as the weight: a pickle must not write it, and a deepcopy -- an EMA
model, a frozen reference -- must not share or duplicate it.
"""

import copy
import pickle
import unittest

import torch

import jittor as jt
from _helpers import capability as _test_capability
from jittor.nn.backends import cudnn as _cudnn


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "No CUDA found")
@_test_capability.library_required("cudnn", backend=jt)
class TestConvFilterCache(unittest.TestCase):
    def test_copies_of_the_weight_leave_the_cache_behind(self):
        conv = torch.nn.Conv2d(8, 8, 3, padding=1).cuda().half()
        x = torch.randn(1, 8, 6, 6, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            want = conv(x).float().cpu().numpy()
        self.assertIn(_cudnn._FILTER_CACHE, conv.weight.__dict__)
        for copied in (pickle.loads(pickle.dumps(conv.weight)), copy.deepcopy(conv.weight)):
            self.assertNotIn(_cudnn._FILTER_CACHE, copied.__dict__)
        twin = copy.deepcopy(conv)
        with torch.no_grad():
            got = twin(x).float().cpu().numpy()
        self.assertEqual(got.tolist(), want.tolist())


if __name__ == "__main__":
    unittest.main()
