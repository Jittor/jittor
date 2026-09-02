# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Device selection inside one process.

``jt.flags.device_id`` / ``jt.set_device`` pick the CUDA device new Vars are
placed on, every Var carries ``device_id``, ops run on their inputs' device,
and ``Var.to_device`` moves data between devices with gradients flowing back.
"""
import unittest

import numpy as np

import jittor as jt


def _count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


class _DeviceCase(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.current_device())
        jt.flags.use_cuda = 1
        jt.set_device(0)

    def tearDown(self):
        if self._saved[1] >= 0:
            jt.set_device(self._saved[1])
        jt.flags.use_cuda = self._saved[0]


@unittest.skipIf(not jt.has_cuda or _count() < 1, "No cuda found")
class TestDeviceSelection(_DeviceCase):
    def test_current_device_and_placement(self):
        self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.flags.device_id, 0)
        x = jt.array(np.ones(4, "float32"))
        self.assertEqual(x.device_id, 0)
        y = x + 1
        self.assertEqual(y.device_id, 0)
        np.testing.assert_array_equal(y.numpy(), np.full(4, 2.0))

    def test_invalid_device_rejected(self):
        with self.assertRaises(Exception):
            jt.set_device(_count() + 5)
        self.assertEqual(jt.current_device(), 0)

    def test_scope_restores_device(self):
        with jt.flag_scope(device_id=0):
            self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.current_device(), 0)


@unittest.skipIf(not jt.has_cuda or _count() < 2, "Needs two CUDA devices")
class TestMultiDevice(_DeviceCase):
    def test_compute_on_second_device(self):
        with jt.flag_scope(device_id=1):
            self.assertEqual(jt.current_device(), 1)
            x = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
            self.assertEqual(x.device_id, 1)
            y = (x * 2 + 1).sum(1)
            self.assertEqual(y.device_id, 1)
            np.testing.assert_allclose(
                y.numpy(), (np.arange(12).reshape(3, 4) * 2 + 1).sum(1))
        self.assertEqual(jt.current_device(), 0)

    def test_transfer_between_devices(self):
        a = jt.array(np.random.RandomState(0).randn(64, 32).astype("float32"))
        b = a.to_device(1)
        self.assertEqual(b.device_id, 1)
        c = jt.matmul(b, b.transpose())
        self.assertEqual(c.device_id, 1)
        d = c.to_device(0)
        self.assertEqual(d.device_id, 0)
        ref = a.numpy() @ a.numpy().T
        np.testing.assert_allclose(c.numpy(), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(d.numpy(), ref, rtol=1e-4, atol=1e-4)

    def test_pending_scalar_follows_operand(self):
        with jt.flag_scope(device_id=1):
            x = jt.ones((5,), "float32")
        # The 3 and the 1 are built on device 0 but have no data yet, so
        # they follow x, as a CPU scalar does in torch.
        y = x * 3 + 1
        self.assertEqual(y.device_id, 1)
        np.testing.assert_array_equal(y.numpy(), np.full(5, 4.0))

    def test_mixed_devices_rejected(self):
        x = jt.array(np.ones(4, "float32"))
        x.sync()
        with jt.flag_scope(device_id=1):
            y = jt.array(np.ones(4, "float32"))
            y.sync()
        with self.assertRaises(Exception):
            (x + y).sync()

    def test_gradient_through_device_copy(self):
        x = jt.array(np.random.RandomState(1).randn(8).astype("float32"))
        y = (x.to_device(1) ** 2).sum()
        self.assertEqual(y.device_id, 1)
        g = jt.grad(y, x)
        self.assertEqual(g.device_id, 0)
        np.testing.assert_allclose(g.numpy(), 2 * x.numpy(), rtol=1e-5)

    def test_library_ops_on_second_device(self):
        # cuBLAS and cuDNN hold one handle per device.
        with jt.flag_scope(device_id=1):
            x = jt.array(np.random.RandomState(2).randn(2, 3, 8, 8).astype("float32"))
            w = jt.array(np.random.RandomState(3).randn(4, 3, 3, 3).astype("float32"))
            y = jt.nn.conv2d(x, w, None, 1, 1)
            self.assertEqual(y.device_id, 1)
            got = y.numpy()
        with jt.flag_scope(use_cuda=0):
            ref = jt.nn.conv2d(jt.array(x.numpy()), jt.array(w.numpy()), None, 1, 1).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    def test_random_on_second_device(self):
        with jt.flag_scope(device_id=1):
            r = jt.rand(1000)
            self.assertEqual(r.device_id, 1)
            v = r.numpy()
        self.assertTrue(0.0 <= v.min() and v.max() <= 1.0 and 0.3 < v.mean() < 0.7)


if __name__ == "__main__":
    unittest.main()
