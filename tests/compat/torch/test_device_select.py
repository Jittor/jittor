"""torch.cuda device selection through the shim.

Mirrors the torch API surface: set_device/current_device/device_count, the
``torch.cuda.device`` context, ``device="cuda:N"`` factories, ``Tensor.device``
with a real index, ``.to("cuda:N")``/``.cuda(N)`` copies, ``Module.to`` in
place, and the mixed-device error.
"""
import unittest

import numpy as np

import jittor as jt
import torch


def _two_devices():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


@unittest.skipIf(not _two_devices(), "Needs two visible CUDA devices")
class TestDeviceSelect(unittest.TestCase):
    def setUp(self):
        torch.cuda.set_device(0)

    def tearDown(self):
        torch.cuda.synchronize()
        torch.cuda.set_device(0)

    def test_set_and_query_current_device(self):
        self.assertEqual(torch.cuda.current_device(), 0)
        torch.cuda.set_device(1)
        self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.device_count(), jt.core.get_device_count())
        torch.cuda.set_device("cuda:0")
        self.assertEqual(torch.cuda.current_device(), 0)
        torch.cuda.set_device(torch.device("cuda:1"))
        self.assertEqual(torch.cuda.current_device(), 1)

    def test_device_context_restores(self):
        with torch.cuda.device(1):
            self.assertEqual(torch.cuda.current_device(), 1)
            x = torch.ones(3)
            self.assertEqual(x.device, torch.device("cuda:1"))
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_factories_take_an_index(self):
        x = torch.randn(4, 4, device="cuda:1")
        self.assertEqual(x.device.index, 1)
        self.assertEqual(x.device, torch.device("cuda", 1))
        self.assertEqual(x.get_device(), 1)
        y = torch.zeros(4, 4, device=torch.device("cuda:1"))
        z = x @ y + 1
        self.assertEqual(z.device.index, 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        w = torch.ones(2)
        self.assertEqual(w.device.index, 0)

    def test_to_and_cuda_copy_across_devices(self):
        a = np.random.RandomState(0).randn(6).astype("float32")
        x = torch.from_numpy(a).cuda()
        self.assertEqual(x.device.index, 0)
        x1 = x.to("cuda:1")
        self.assertEqual(x1.device.index, 1)
        x1b = x.cuda(1)
        self.assertEqual(x1b.device.index, 1)
        back = x1.to(torch.device("cuda:0"))
        self.assertEqual(back.device.index, 0)
        np.testing.assert_array_equal(back.cpu().numpy(), a)
        np.testing.assert_array_equal((x1 * 2).cpu().numpy(), a * 2)
        # .to("cuda") without an index means the current device
        torch.cuda.set_device(1)
        self.assertEqual(x.to("cuda").device.index, 1)

    def test_module_to_moves_parameters_in_place(self):
        layer = torch.nn.Linear(8, 4)
        w = layer.weight
        layer.to("cuda:1")
        self.assertIs(layer.weight, w)
        self.assertEqual(layer.weight.device.index, 1)
        x = torch.randn(2, 8, device="cuda:1")
        out = layer(x)
        self.assertEqual(out.device.index, 1)
        out.sum().backward()
        self.assertEqual(layer.weight.grad.device.index, 1)
        ref = x.cpu().numpy() @ layer.weight.detach().cpu().numpy().T + layer.bias.detach().cpu().numpy()
        np.testing.assert_allclose(out.detach().cpu().numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_mixed_devices_raise(self):
        x = torch.ones(4, device="cuda:0")
        y = torch.ones(4, device="cuda:1")
        with self.assertRaises(Exception):
            (x + y).sum().item()

    def test_device_of_and_properties(self):
        x = torch.ones(2, device="cuda:1")
        with torch.cuda.device_of(x):
            self.assertEqual(torch.cuda.current_device(), 1)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.assertTrue(torch.cuda.get_device_name(1))


if __name__ == "__main__":
    unittest.main()
