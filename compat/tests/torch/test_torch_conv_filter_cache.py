"""A half-precision convolution's moved weight copies and pickles as any other.

Without a backward the cuDNN path moves a half-precision filter into OHWI
storage (``jittor/nn/backends/cudnn.py``): same values, other strides. A pickle,
a deepcopy -- an EMA model, a frozen reference -- and a state dict must see
the weight, not its layout, and must not carry the version note the move keeps.
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
class TestConvFilterLayout(unittest.TestCase):
    def test_a_moved_weight_copies_pickles_and_reloads(self):
        conv = torch.nn.Conv2d(8, 8, 3, padding=1).cuda().half()
        x = torch.randn(1, 8, 6, 6, device="cuda", dtype=torch.float16)
        values = conv.weight.detach().cpu().numpy()
        with torch.no_grad():
            conv(x)
            self.assertIn(_cudnn._FILTER_OHWI, conv.weight.__dict__)
            want = conv(x).float().cpu().numpy()
        self.assertTrue(_cudnn._is_ohwi_storage(conv.weight))
        for copied in (pickle.loads(pickle.dumps(conv.weight)), copy.deepcopy(conv.weight)):
            self.assertNotIn(_cudnn._FILTER_OHWI, copied.__dict__)
            self.assertEqual(copied.detach().cpu().numpy().tolist(), values.tolist())
        twin = torch.nn.Conv2d(8, 8, 3, padding=1).cuda().half()
        twin.load_state_dict(conv.state_dict())
        with torch.no_grad():
            got = copy.deepcopy(conv)(x).float().cpu().numpy()
            reloaded = twin(x).float().cpu().numpy()
        self.assertEqual(got.tolist(), want.tolist())
        self.assertEqual(reloaded.tolist(), want.tolist())


if __name__ == "__main__":
    unittest.main()
