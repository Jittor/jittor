# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""torch's device API under the Jittor facade: real device indices.

``torch.cuda.set_device`` switches the device new tensors are created on,
``Tensor.device`` reports the index a tensor really lives on, ``.to("cuda:N")``
copies across devices, ``device="cuda:N"`` builds on N, and the device
context managers scope the default device.
"""
import unittest

import numpy as np

import jittor as jt

try:
    import torch
    _IS_SHIM = getattr(torch, "__name__", "") == "jittor" or hasattr(
        torch, "_torch_compat_install_context")
except Exception:  # pragma: no cover - no facade deployed
    torch = None
    _IS_SHIM = False


def _count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


class _Case(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.current_device())
        jt.flags.use_cuda = 1
        torch.cuda.set_device(0)

    def tearDown(self):
        if self._saved[1] >= 0:
            jt.set_device(self._saved[1])
        jt.flags.use_cuda = self._saved[0]


@unittest.skipIf(not _IS_SHIM or not jt.has_cuda or _count() < 1, "needs the torch facade and CUDA")
class TestDeviceApi(_Case):
    def test_count_current_and_properties(self):
        self.assertEqual(torch.cuda.device_count(), _count())
        self.assertEqual(torch.cuda.current_device(), 0)
        x = torch.zeros(3, device="cuda")
        self.assertEqual(x.device, torch.device("cuda", 0))
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(torch.zeros(2).cpu().get_device(), -1)

    def test_invalid_device_rejected(self):
        with self.assertRaises(RuntimeError):
            torch.cuda.set_device(_count() + 3)


@unittest.skipIf(not _IS_SHIM or not jt.has_cuda or _count() < 2, "needs two CUDA devices")
class TestMultiDevice(_Case):
    def test_set_device_places_new_tensors(self):
        torch.cuda.set_device(1)
        try:
            self.assertEqual(torch.cuda.current_device(), 1)
            x = torch.ones(4)
            self.assertEqual(str(x.device), "cuda:1")
        finally:
            torch.cuda.set_device(0)
        self.assertEqual(str(torch.ones(1).device), "cuda:0")

    def test_factory_device_index(self):
        x = torch.zeros(3, 2, device="cuda:1")
        self.assertEqual(x.device, torch.device("cuda:1"))
        self.assertEqual(x.get_device(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        y = torch.full((2,), 7.0, device=torch.device("cuda", 1))
        self.assertEqual(y.device.index, 1)
        np.testing.assert_array_equal(y.cpu().numpy(), np.full(2, 7.0))

    def test_to_and_cuda_with_index(self):
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertEqual(a.device.index, 0)
        b = a.to("cuda:1")
        self.assertEqual(b.device.index, 1)
        np.testing.assert_array_equal(b.cpu().numpy(), a.cpu().numpy())
        self.assertIs(b.to("cuda:1"), b)
        self.assertEqual(b.to("cuda").device.index, 1)
        c = b.cuda(0)
        self.assertEqual(c.device.index, 0)
        d = a.to(b)
        self.assertEqual(d.device.index, 1)

    def test_compute_on_second_device(self):
        a = torch.randn(16, 8).to("cuda:1")
        y = (a @ a.t()).sum()
        self.assertEqual(y.device.index, 1)
        ref = a.cpu().numpy()
        self.assertAlmostEqual(float(y), float((ref @ ref.T).sum()), places=2)

    def test_device_contexts(self):
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.current_device(), 1)
            z = torch.ones(2)
            self.assertEqual(z.device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        with torch.device("cuda:1"):
            w = torch.randn(4)
            self.assertEqual(w.device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        with torch.cuda.device_of(w):
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_mixed_devices_rejected(self):
        a = torch.ones(3)
        b = torch.ones(3, device="cuda:1")
        with self.assertRaises(RuntimeError):
            (a + b).sum().item()

    def test_module_on_second_device(self):
        m = torch.nn.Linear(4, 2).to("cuda:1")
        self.assertEqual(m.weight.device.index, 1)
        x = torch.randn(3, 4, device="cuda:1")
        out = m(x)
        self.assertEqual(out.device.index, 1)
        out.sum().backward()
        self.assertEqual(m.weight.grad.device.index, 1)
        ref = x.cpu().numpy() @ m.weight.detach().cpu().numpy().T + m.bias.detach().cpu().numpy()
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
