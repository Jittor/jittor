# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Placement: one process, every visible CUDA device.

Every Var carries ``device_id``, the device it lives on or will be computed
on.  ``jt.set_device(i)`` / ``jt.flags.device_id`` pick the device new Vars go
to -- in place, no process restart.  An op runs where its inputs are, and
mixing two devices in one op is refused at graph-construction time, as torch
refuses it.

Reading ``x.device_id`` only says what jittor *believes*.
``_pointer_device`` below asks the CUDA driver where the bytes actually are,
which is the claim these tests have to make: the second device is really in
use, not merely recorded.
"""
import ctypes
import unittest

import numpy as np

import jittor as jt


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


_HAS_CUDA = jt.has_cuda and _device_count() >= 1
_TWO_DEVICES = jt.has_cuda and _device_count() >= 2

# CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, from cuda.h. The driver API is used
# rather than the runtime's cudaPointerGetAttributes because that struct's
# layout changed between CUDA versions while cuPointerGetAttribute's signature
# has not, and libcuda.so.1 ships with the driver, so it is there whenever a
# GPU is. Get the number right: 15 is IS_GPU_DIRECT_RDMA_CAPABLE, which
# answers 0 for every pointer and so reports every tensor on device 0.
_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
_libcuda = None


def _pointer_device(ptr):
    """The device index the driver says ``ptr`` is allocated on, or None.

    ``ptr`` must be a device pointer (``Var.device_raw_ptr``). Returns None
    when the driver library cannot be loaded or the query fails, so a test can
    say so instead of silently passing.
    """
    global _libcuda
    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            _libcuda = False
    if _libcuda is False:
        return None
    value = ctypes.c_int(-1)
    res = _libcuda.cuPointerGetAttribute(
        ctypes.byref(value), _CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ctypes.c_void_p(ptr))
    if res != 0:
        return None
    return value.value


class _DeviceCase(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.current_device())
        jt.flags.use_cuda = 1
        jt.set_device(0)

    def tearDown(self):
        jt.sync_all(True)
        if self._saved[1] >= 0:
            jt.set_device(self._saved[1])
        jt.flags.use_cuda = self._saved[0]


@unittest.skipIf(not _HAS_CUDA, "No cuda found")
class TestCurrentDevice(_DeviceCase):
    def test_current_device_is_the_flag(self):
        self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.flags.device_id, 0)

    def test_new_vars_take_the_current_device(self):
        x = jt.array(np.ones(4, "float32"))
        self.assertEqual(x.device_id, 0)
        self.assertEqual((x + 1).device_id, 0)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(Exception):
            jt.set_device(_device_count() + 5)
        # ... and the current device is unchanged, not left half-switched.
        self.assertEqual(jt.current_device(), 0)

    def test_set_device_does_not_restart_the_process(self):
        # The old setter re-exec'd the interpreter with CUDA_VISIBLE_DEVICES
        # rewritten, so everything built before the switch was gone. Nothing
        # may be lost across a switch now.
        marker = jt.array(np.array([42.0], "float32"))
        marker.sync()
        jt.set_device(0)
        self.assertEqual(float(marker.numpy()[0]), 42.0)


@unittest.skipIf(not _TWO_DEVICES, "Needs two visible CUDA devices")
class TestSecondDevice(_DeviceCase):
    def test_data_really_lands_on_the_second_device(self):
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones(1024, "float32"))
            x.sync()
            self.assertEqual(x.device_id, 1)
            where = _pointer_device(x.device_raw_ptr)
        self.assertIsNotNone(where, "cuPointerGetAttribute unavailable")
        self.assertEqual(where, 1)
        y = jt.array(np.ones(1024, "float32"))
        y.sync()
        self.assertEqual(_pointer_device(y.device_raw_ptr), 0)

    def test_compute_runs_on_the_second_device(self):
        a = np.arange(12, dtype="float32").reshape(3, 4)
        with jt.flag_scope(device_id=1):
            x = jt.array(a)
            y = (x * 2 + 1).sum(1)
            self.assertEqual(y.device_id, 1)
            y.sync()
            self.assertEqual(_pointer_device(y.device_raw_ptr), 1)
            got = y.numpy()
        np.testing.assert_allclose(got, (a * 2 + 1).sum(1))
        # the scope gave the caller's device back
        self.assertEqual(jt.current_device(), 0)

    def test_scope_restores_the_current_device(self):
        with jt.flag_scope(device_id=1):
            self.assertEqual(jt.current_device(), 1)
        self.assertEqual(jt.current_device(), 0)
        self.assertEqual(jt.flags.device_id, 0)

    def test_pending_scalar_follows_its_operand(self):
        # The 3 and the 1 are built while device 0 is current, but they are
        # pending one-element sources with no data anywhere -- torch's CPU
        # scalars. They follow x rather than making this an error.
        with jt.flag_scope(device_id=1):
            x = jt.ones((5,), "float32")
        y = x * 3 + 1
        self.assertEqual(y.device_id, 1)
        np.testing.assert_array_equal(y.numpy(), np.full(5, 4.0))

    def test_a_placed_pending_tensor_is_not_retargeted(self):
        # This is the case the scalar exemption must NOT cover, and it is why
        # pendingness alone is not the rule: `big` was deliberately placed on
        # device 0 and merely has not been executed yet. Letting it follow
        # whatever it meets would move a user's data to another device without
        # a word, where torch raises.
        big = jt.array(np.ones(1000, "float32"))
        self.assertEqual(big.device_id, 0)
        with jt.flag_scope(device_id=1):
            other = jt.array(np.ones(1000, "float32"))
        with self.assertRaises(Exception):
            (big + other).sync()

    def test_a_pending_broadcast_constant_does_follow(self):
        # The documented edge of the rule. jt.zeros(n) / jt.ones(n) are
        # `unary(0).broadcast(n)`: a one-element constant with the _is_scalar
        # flag carried through the broadcast, holding no data anywhere until
        # it runs. So it follows its operand exactly as the `3` in `x * 3`
        # does, even though the caller named it. Nothing is lost: a constant
        # produced on the other device is bit-identical, and every path that
        # actually carries data (jt.array of more than one element, or any
        # value already computed) is refused by the two tests above.
        with jt.flag_scope(device_id=1):
            x = jt.array(np.ones(1000, "float32"))
        z = jt.zeros((1000,), "float32")
        self.assertEqual(z.device_id, 0)
        both = z + x
        self.assertEqual(both.device_id, 1)
        np.testing.assert_array_equal(both.numpy(), np.ones(1000, "float32"))

    def test_a_one_element_tensor_is_not_a_scalar(self):
        # Element count alone would exempt this; it is a real user tensor
        # placed on device 0 and must be refused just like the 1000-element
        # one above.
        one = jt.array(np.ones(1, "float32"))
        one.sync()
        with jt.flag_scope(device_id=1):
            other = jt.array(np.ones(1, "float32"))
            other.sync()
        with self.assertRaises(Exception):
            (one + other).sync()

    def test_mixed_devices_are_refused_with_torch_s_message(self):
        x = jt.array(np.ones(4, "float32"))
        x.sync()
        with jt.flag_scope(device_id=1):
            y = jt.array(np.ones(4, "float32"))
            y.sync()
        with self.assertRaises(Exception) as caught:
            (x + y).sync()
        self.assertIn("same CUDA device", str(caught.exception))

    def test_backward_stays_on_the_forward_s_device(self):
        # jt.grad builds new ops; they have to follow the same rule, or a
        # forward that was checked would be followed by a silently mixed
        # backward.
        rng = np.random.RandomState(3)
        w = rng.randn(32, 16).astype("float32")
        x = rng.randn(8, 32).astype("float32")
        with jt.flag_scope(device_id=1):
            wv = jt.array(w)
            xv = jt.array(x)
            loss = (jt.matmul(xv, wv) ** 2).sum()
            self.assertEqual(loss.device_id, 1)
            gw = jt.grad(loss, wv)
            self.assertEqual(gw.device_id, 1)
            gw.sync()
            self.assertEqual(_pointer_device(gw.device_raw_ptr), 1)
            got = gw.numpy()
        np.testing.assert_allclose(got, 2 * x.T @ (x @ w), rtol=1e-3, atol=1e-3)

    def test_backward_of_a_mixed_graph_is_refused_too(self):
        a = jt.array(np.ones((4, 4), "float32"))
        a.sync()
        with jt.flag_scope(device_id=1):
            b = jt.array(np.ones((4, 4), "float32"))
            b.sync()
        with self.assertRaises(Exception):
            jt.grad((a * b).sum(), a).sync()

    def test_cudnn_and_cublas_have_a_handle_per_device(self):
        # A cuDNN/cuBLAS handle only works on the device it was created on, so
        # this is what catches a missing handle swap.
        rng = np.random.RandomState(2)
        x = rng.randn(2, 3, 8, 8).astype("float32")
        w = rng.randn(4, 3, 3, 3).astype("float32")
        with jt.flag_scope(device_id=1):
            y = jt.nn.conv2d(jt.array(x), jt.array(w), None, 1, 1)
            self.assertEqual(y.device_id, 1)
            got_conv = y.numpy()
            m = jt.matmul(jt.array(x.reshape(6, 64)), jt.array(x.reshape(6, 64)).transpose())
            got_mm = m.numpy()
        with jt.flag_scope(use_cuda=0):
            ref_conv = jt.nn.conv2d(jt.array(x), jt.array(w), None, 1, 1).numpy()
        np.testing.assert_allclose(got_conv, ref_conv, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(
            got_mm, x.reshape(6, 64) @ x.reshape(6, 64).T, rtol=1e-3, atol=1e-3)

    def test_curand_has_a_generator_per_device(self):
        with jt.flag_scope(device_id=1):
            r = jt.rand(4096)
            self.assertEqual(r.device_id, 1)
            r.sync()
            self.assertEqual(_pointer_device(r.device_raw_ptr), 1)
            v = r.numpy()
        self.assertTrue(0.0 <= v.min() and v.max() <= 1.0)
        self.assertTrue(0.4 < v.mean() < 0.6, v.mean())

    def test_both_devices_in_one_run(self):
        # Two independent graphs in one sync: the executor has to switch per
        # op and wait on both devices at the end, not only on the current one.
        a = np.random.RandomState(4).randn(128, 128).astype("float32")
        x0 = jt.array(a)
        y0 = jt.matmul(x0, x0)
        with jt.flag_scope(device_id=1):
            x1 = jt.array(a)
            y1 = jt.matmul(x1, x1)
        jt.sync([y0, y1], device_sync=True)
        np.testing.assert_allclose(y0.numpy(), y1.numpy(), rtol=1e-4, atol=1e-3)
        self.assertEqual(y0.device_id, 0)
        self.assertEqual(y1.device_id, 1)


if __name__ == "__main__":
    unittest.main()
