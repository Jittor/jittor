"""Real ACL peer copies must preserve values across two NPU indices."""
import unittest

import numpy as np
import jittor as jt
import torch
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestNpuPeerCopy(unittest.TestCase):
    def test_device_generated_int64_round_trip(self):
        if torch.npu.device_count() < 2:
            self.skipTest("requires two visible NPUs")
        self.assertTrue(hasattr(torch, "_torch_compat_install_context"))
        original_device = torch.npu.current_device()
        self.addCleanup(torch.npu.set_device, original_device)
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"):
            before = jt.core.backend_fallback_count()
            torch.npu.set_device(0)
            source = torch.arange(8, dtype=torch.int64, device="npu:0").reshape(2, 4)
            on_one = source.to("npu:1")
            back = on_one.to("npu:0")
            self.assertEqual(str(on_one.device), "npu:1")
            self.assertEqual(str(back.device), "npu:0")
            on_one.sync()
            back.sync()
            self.assertEqual(on_one.location(), "device")
            self.assertEqual(back.location(), "device")
            expected = np.arange(8, dtype=np.int64).reshape(2, 4)
            np.testing.assert_array_equal(on_one.detach().cpu().numpy(), expected)
            np.testing.assert_array_equal(back.detach().cpu().numpy(), expected)
            self.assertEqual(jt.core.backend_fallback_count(), before)


if __name__ == "__main__":
    unittest.main()
