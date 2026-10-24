"""FSDP internal pad follows the gradient even inside CPU-default frontend."""
from types import SimpleNamespace
import os
import unittest

import numpy as np
import jittor as jt
import torch

from jittor.compat.fsdp2 import common
from jittor._runtime.fallback import forbid_backend_fallbacks


class TestFSDP2PaddingPlacement(unittest.TestCase):
    def test_internal_padding_forward_and_gradient(self):
        kind = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        self.assertIn(kind, ("cpu", "npu", "cuda"))
        ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
        target = kind if kind == "cpu" else kind + ":" + str(ordinal)
        if kind != "cpu":
            self.assertTrue(getattr(torch, kind).is_available())
            getattr(torch, kind).set_device(ordinal)
        state = SimpleNamespace(frontend_type=torch.Tensor)

        def resident(value):
            value.sync()
            self.assertEqual(value.device.type, kind)
            self.assertEqual(value.location(), "cpu" if kind == "cpu" else "device")
            if kind != "cpu":
                self.assertEqual(value.device_id, ordinal)

        with jt.runtime.scope(use_cuda=0), common._frontend_scope(state), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            for length, padded in ((1, 2), (3, 4), (5, 8), (8, 8)):
                with self.subTest(length=length, padded=padded):
                    x = torch.tensor(np.arange(length, dtype=np.float32), device=target, requires_grad=True)
                    y = common._pad_flat(x, padded)
                    weights = torch.tensor(np.arange(1, padded + 1, dtype=np.float32), device=target)
                    loss = (y * weights).sum()
                    loss.backward()
                    for value in (x, y, loss, x.grad):
                        resident(value)
                    expected = np.pad(np.arange(length, dtype=np.float32), (0, padded - length))
                    np.testing.assert_array_equal(y.detach().cpu().numpy(), expected)
                    np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), np.arange(1, length + 1))
                    for default in (jt.zeros(1), torch.zeros(1), torch.zeros_like(x, device="cpu")):
                        default.sync()
                        self.assertEqual(default.location(), "cpu", "internal pad must not alter caller factories")
                    self.assertEqual(jt.runtime.use_cuda, 0)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()
