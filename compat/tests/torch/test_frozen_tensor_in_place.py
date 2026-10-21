"""An in-place write leaves ``requires_grad`` where it was.

``requires_grad_(False)`` need not set the native stop-grad bit, and the
in-place paths read that bit alone to decide what to restore after assigning:
a frozen tensor written under ``no_grad`` -- ``copy_``, ``add_``, ``x.data =``,
``x.data[i] =`` -- came back trainable.
"""
import unittest

import torch


def _write(tensor, how):
    with torch.no_grad():
        if how == "copy_":
            tensor.copy_(torch.zeros(3, device=tensor.device))
        elif how == "add_":
            tensor.add_(1.0)
        elif how == "data":
            tensor.data = torch.zeros(3, device=tensor.device)
        else:
            tensor.data[0] = 5.0


class TestFrozenTensorInPlace(unittest.TestCase):
    def _check(self, device):
        for how in ("copy_", "add_", "data", "data_item"):
            frozen = torch.ones(3, device=device, requires_grad=True)
            frozen.requires_grad_(False)
            _write(frozen, how)
            self.assertFalse(frozen.requires_grad, how)
            trainable = torch.ones(3, device=device, requires_grad=True)
            _write(trainable, how)
            self.assertTrue(trainable.requires_grad, how)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
    def test_cuda(self):
        self._check("cuda")


if __name__ == "__main__":
    unittest.main()
