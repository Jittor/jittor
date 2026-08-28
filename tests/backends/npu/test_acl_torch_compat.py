import unittest

import numpy as np

import jittor as torch
import jittor as jt


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestACLTorchCompat(unittest.TestCase):
    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_empty_cuda_tensor(self):
        device = torch.device("cuda")
        empty = torch.tensor([], dtype=torch.float32, device=device)
        self.assertEqual(empty.numel(), 0)
        self.assertTrue(empty.is_cuda)

        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        joined = torch.cat((empty, value))
        np.testing.assert_array_equal(joined.cpu().numpy(), [1.0])

    def test_default_device_follows_execution_flag(self):
        with jt.flag_scope(use_acl=0, use_cuda=0):
            self.assertEqual(torch.get_default_device().type, "cpu")
        with jt.flag_scope(use_acl=1, use_cuda=1):
            self.assertEqual(torch.get_default_device().type, "cuda")
