# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Per-Var CUDA device placement in one process.

``jt.flags.device_id`` is the current device: new Vars are placed on it and
it becomes the CUDA current device, without restarting the process. Every Var
carries its device (``Var.device_index()``), ops run on their inputs' device,
``Var.to_device`` copies across devices, and mixing devices in one op is an
error -- torch's rules.
"""
import unittest

import numpy as np

import jittor as jt


@unittest.skipIf(not jt.has_cuda, "No cuda found")
@unittest.skipIf(jt.has_cuda and jt.core.get_device_count() < 2, "Needs two visible CUDA devices")
class TestMultiDevice(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.device_id)
        jt.flags.use_cuda = 1
        jt.flags.device_id = 0

    def tearDown(self):
        jt.sync_all(True)
        jt.flags.use_cuda, jt.flags.device_id = self._saved

    def test_current_device_places_new_vars(self):
        a = np.random.RandomState(0).randn(64, 64).astype("float32")
        jt.flags.device_id = 1
        x = jt.array(a)
        self.assertEqual(x.device_index(), 1)
        y = jt.matmul(x, x) + 1
        self.assertEqual(y.device_index(), 1)
        np.testing.assert_allclose(y.numpy(), a @ a + 1, rtol=1e-4, atol=1e-4)
        self.assertEqual(x.location(), "device")
        jt.flags.device_id = 0
        z = jt.array(a)
        self.assertEqual(z.device_index(), 0)

    def test_to_device_round_trip(self):
        a = np.random.RandomState(1).randn(1000).astype("float32")
        x = jt.array(a)
        x1 = x.to_device(1)
        self.assertEqual(x1.device_index(), 1)
        self.assertIs(x1.to_device(1), x1)
        back = x1.to_device(0)
        self.assertEqual(back.device_index(), 0)
        np.testing.assert_array_equal(back.numpy(), a)
        np.testing.assert_array_equal((x1 * 2).numpy(), a * 2)

    def test_mixing_devices_is_an_error(self):
        x = jt.array(np.ones(4, "float32"))
        y = x.to_device(1)
        with self.assertRaises(Exception):
            (x + y).sync()

    def test_scalar_constants_follow_the_data(self):
        a = np.arange(8, dtype="float32")
        x = jt.array(a).to_device(1)
        y = x * 2 + 1
        self.assertEqual(y.device_index(), 1)
        np.testing.assert_array_equal(y.numpy(), a * 2 + 1)

    def test_gradient_flows_back_across_devices(self):
        x = jt.array(np.random.RandomState(2).randn(16).astype("float32"))
        y = (x.to_device(1) * 3).sum()
        self.assertEqual(y.device_index(), 1)
        g = jt.grad(y, x)
        self.assertEqual(g.device_index(), 0)
        np.testing.assert_allclose(g.numpy(), np.full(16, 3.0))

    def test_training_step_on_second_device(self):
        rng = np.random.RandomState(3)
        w = rng.randn(32, 16).astype("float32"); x = rng.randn(8, 32).astype("float32")
        jt.flags.device_id = 1
        wv = jt.array(w); xv = jt.array(x)
        loss = (jt.matmul(xv, wv) ** 2).sum()
        gw = jt.grad(loss, wv)
        self.assertEqual(gw.device_index(), 1)
        expected = 2 * x.T @ (x @ w)
        np.testing.assert_allclose(gw.numpy(), expected, rtol=1e-3, atol=1e-3)

    def test_random_and_reductions_on_second_device(self):
        jt.flags.device_id = 1
        jt.set_global_seed(5)
        r = jt.rand(1000)
        self.assertEqual(r.device_index(), 1)
        m = float(r.mean())
        self.assertTrue(0.4 < m < 0.6, m)
        s = jt.argsort(r)[0]
        self.assertEqual(s.device_index(), 1)
        self.assertEqual(int(s.numel()), 1000)

    def test_both_devices_in_one_step_and_full_sync(self):
        a = np.random.RandomState(4).randn(256, 256).astype("float32")
        x0 = jt.array(a)
        x1 = x0.to_device(1)
        y0 = jt.matmul(x0, x0)
        y1 = jt.matmul(x1, x1)
        jt.sync([y0, y1], device_sync=True)
        np.testing.assert_allclose(y1.to_device(0).numpy(), y0.numpy(), rtol=1e-4, atol=1e-3)

    def test_out_of_range_device_is_rejected(self):
        with self.assertRaises(Exception):
            jt.flags.device_id = 1000
        with self.assertRaises(Exception):
            jt.array(np.ones(2, "float32")).to_device(1000).sync()


if __name__ == "__main__":
    unittest.main()
