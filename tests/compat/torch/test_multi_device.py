# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""torch's device API under the facade, with real indices.

Every one of these used to be a lie in a specific way: ``current_device()``
returned 0 whatever you did, ``set_device(1)`` was refused as unimplemented,
``Tensor.device`` reported ``cuda:0`` for every tensor, ``.to("cuda:1")``
dropped the index, ``device="cuda:1"`` built on device 0, and
``torch.device("cuda:1")`` as a context manager did nothing.
"""
import gc
import unittest

import numpy as np

import jittor as jt

try:
    import torch
    _IS_SHIM = getattr(torch, "__name__", "") == "jittor" or hasattr(
        torch, "_torch_compat_install_context")
except Exception:  # pragma: no cover - facade not deployed
    torch = None
    _IS_SHIM = False


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


_HAS_CUDA = _IS_SHIM and jt.has_cuda and _device_count() >= 1
_TWO_DEVICES = _IS_SHIM and jt.has_cuda and _device_count() >= 2


class _Case(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.current_device())
        jt.flags.use_cuda = 1
        torch.cuda.set_device(0)

    def tearDown(self):
        # The Modules built here hold Vars; without a collection the file is
        # reported as leaking process-wide state it merely has not freed yet.
        gc.collect()
        jt.sync_all(True)
        if self._saved[1] >= 0:
            jt.set_device(self._saved[1])
        jt.flags.use_cuda = self._saved[0]


@unittest.skipIf(not _HAS_CUDA, "needs the torch facade and a CUDA device")
class TestDeviceApi(_Case):
    def test_count_and_current(self):
        self.assertEqual(torch.cuda.device_count(), _device_count())
        self.assertEqual(torch.cuda.current_device(), 0)

    def test_tensor_device_has_an_index(self):
        x = torch.zeros(3, device="cuda")
        self.assertEqual(x.device, torch.device("cuda", 0))
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(torch.zeros(2).cpu().get_device(), -1)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(RuntimeError):
            torch.cuda.set_device(_device_count() + 3)

    def test_default_device_reports_the_current_one(self):
        self.assertEqual(torch.get_default_device(), torch.device("cuda", 0))


@unittest.skipIf(not _TWO_DEVICES, "needs two visible CUDA devices")
class TestMultiDeviceFacade(_Case):
    def test_set_device_places_new_tensors(self):
        torch.cuda.set_device(1)
        try:
            self.assertEqual(torch.cuda.current_device(), 1)
            x = torch.ones(4)
            self.assertEqual(str(x.device), "cuda:1")
        finally:
            torch.cuda.set_device(0)
        self.assertEqual(str(torch.ones(1).device), "cuda:0")

    def test_factory_device_index_creates_there(self):
        x = torch.zeros(3, 2, device="cuda:1")
        self.assertEqual(x.device, torch.device("cuda:1"))
        self.assertEqual(x.get_device(), 1)
        # created on 1 without moving the caller's current device
        self.assertEqual(torch.cuda.current_device(), 0)
        y = torch.full((2,), 7.0, device=torch.device("cuda", 1))
        self.assertEqual(y.device.index, 1)
        np.testing.assert_array_equal(y.cpu().numpy(), np.full(2, 7.0))
        r = torch.randn(4, 4, device="cuda:1")
        self.assertEqual(r.device.index, 1)

    def test_to_and_cuda_with_an_index(self):
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertEqual(a.device.index, 0)
        b = a.to("cuda:1")
        self.assertEqual(b.device.index, 1)
        np.testing.assert_array_equal(b.cpu().numpy(), a.cpu().numpy())
        # already there: no cross-device copy. (The facade's residency
        # helper may still hand back a fresh Var -- .cpu() above can leave
        # `b` host-resident -- so this is about the device, not identity.)
        self.assertEqual(b.to("cuda:1").device.index, 1)
        # a bare "cuda" leaves it where it is, as in torch
        self.assertEqual(b.to("cuda").device.index, 1)
        c = b.cuda(0)
        self.assertEqual(c.device.index, 0)
        np.testing.assert_array_equal(c.cpu().numpy(), a.cpu().numpy())
        # .to(other_tensor) takes the other's device. Use a tensor that has
        # not been through .cpu(): the facade's residency model makes a
        # host-resident Var report device "cpu" whatever its index.
        other = torch.ones(3, device="cuda:1")
        d = torch.arange(3, dtype=torch.float32).to(other)
        self.assertEqual(d.device.index, 1)

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

    def test_accelerator_follows(self):
        torch.accelerator.set_device_index(1)
        try:
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        finally:
            torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_set_default_device_with_an_index(self):
        torch.set_default_device("cuda:1")
        try:
            self.assertEqual(torch.get_default_device().index, 1)
            self.assertEqual(torch.ones(3).device.index, 1)
        finally:
            torch.set_default_device("cuda:0")
        self.assertEqual(torch.ones(3).device.index, 0)

    def test_compute_on_the_second_device(self):
        a = torch.randn(16, 8).to("cuda:1")
        y = (a @ a.t()).sum()
        self.assertEqual(y.device.index, 1)
        ref = a.cpu().numpy()
        self.assertAlmostEqual(float(y), float((ref @ ref.T).sum()), places=1)

    def test_mixed_devices_are_refused(self):
        a = torch.ones(3, device="cuda:0")
        b = torch.ones(3, device="cuda:1")
        with self.assertRaises(Exception):
            float((a + b).sum())

    def test_module_to_moves_parameters_in_place(self):
        layer = torch.nn.Linear(8, 4)
        w = layer.weight
        layer.to("cuda:1")
        # torch's Module.to is in place: the Parameter object survives, so an
        # optimizer built before the move still holds the right object.
        self.assertIs(layer.weight, w)
        self.assertEqual(layer.weight.device.index, 1)
        x = torch.randn(2, 8, device="cuda:1")
        out = layer(x)
        self.assertEqual(out.device.index, 1)
        out.sum().backward()
        self.assertEqual(layer.weight.grad.device.index, 1)
        ref = (x.cpu().numpy() @ layer.weight.detach().cpu().numpy().T
               + layer.bias.detach().cpu().numpy())
        np.testing.assert_allclose(
            out.detach().cpu().numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_module_cuda_with_an_index(self):
        layer = torch.nn.Linear(4, 2)
        w = layer.weight
        layer.cuda(1)
        self.assertIs(layer.weight, w)
        self.assertEqual(layer.weight.device.index, 1)


if __name__ == "__main__":
    unittest.main()
