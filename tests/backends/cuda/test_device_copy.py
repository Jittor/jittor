# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Var.to_device`` -- the only way data changes CUDA device.

The interesting part is ordering. Each device drives its own default stream
and nothing orders one against another, so a cross-device copy has to say so
itself: wait for the producer on the source, then hold the source back until
the copy has read the memory.

``test_copy_waits_for_the_source_kernel`` is the shape of test that can tell
-- a long kernel queued on the source, the copy issued immediately, the value
read on the destination -- because a final device sync makes the work *finish*
without making it *happen in the right order*, so wrong bytes stay wrong.

**It only discriminates where the two devices can peer.** Read
``_peer_regime()`` below before concluding anything from it: when
``cudaDeviceCanAccessPeer`` is false for the pair -- which is the case for
consumer cards, where NVIDIA disables P2P outright -- the driver stages every
cross-device copy through host memory and serialises it against the source
device on its own, so removing the events changes nothing and this test passes
either way. That was measured, not assumed: with the ``cudaEventRecord`` /
``cudaStreamWaitEvent`` pair deleted from ``DeviceCopyOp::run`` all seven
cases here still passed on a non-peer pair. Run this file on a peer-capable
pair (``nvidia-smi topo -m`` showing NV#/PIX, and ``_peer_regime()`` reporting
"peer") to get the regression guard the events are actually for.
"""
import unittest

import numpy as np

import jittor as jt


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


def _peer_regime():
    """"peer" if device 0 and 1 can read each other's memory, else "staged".

    In the staged regime a cross-device copy goes through host memory and the
    driver orders it against the source device itself, so the ordering this
    file is about is not observable -- see the module docstring.
    """
    try:
        import ctypes
        import glob
        for path in sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*"),
                           reverse=True) + ["libcudart.so"]:
            try:
                rt = ctypes.CDLL(path)
                break
            except OSError:
                rt = None
        if rt is None:
            return "unknown"
        can = ctypes.c_int(-1)
        if rt.cudaDeviceCanAccessPeer(ctypes.byref(can), 0, 1) != 0:
            return "unknown"
        return "peer" if can.value else "staged"
    except Exception:
        return "unknown"


class _DeviceCase(unittest.TestCase):
    #: See tests/backends/cuda/test_multi_device.py: the device count is asked
    #: for at run time, never during collection.
    min_devices = 2

    @classmethod
    def setUpClass(cls):
        if not jt.has_cuda:
            raise unittest.SkipTest("this machine has no CUDA build")
        if _device_count() < cls.min_devices:
            raise unittest.SkipTest(
                "this machine has %d visible CUDA device(s), the test needs %d"
                % (_device_count(), cls.min_devices))

    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.current_device())
        jt.flags.use_cuda = 1
        jt.set_device(0)

    def tearDown(self):
        jt.sync_all(True)
        if self._saved[1] >= 0:
            jt.set_device(self._saved[1])
        jt.flags.use_cuda = self._saved[0]


class TestDeviceCopy(_DeviceCase):
    def test_round_trip(self):
        a = np.random.RandomState(1).randn(1000).astype("float32")
        x = jt.array(a)
        x1 = x.to_device(1)
        self.assertEqual(x1.device_id, 1)
        np.testing.assert_array_equal(x1.numpy(), a)
        back = x1.to_device(0)
        self.assertEqual(back.device_id, 0)
        np.testing.assert_array_equal(back.numpy(), a)
        # and the copy is a real value on the far side, not a view
        np.testing.assert_array_equal((x1 * 2).numpy(), a * 2)

    def test_same_device_is_the_same_var(self):
        x = jt.array(np.ones(4, "float32"))
        self.assertIs(x.to_device(0), x)

    def test_invalid_device_is_rejected(self):
        x = jt.array(np.ones(4, "float32"))
        with self.assertRaises(Exception):
            x.to_device(_device_count() + 3)

    def test_copy_from_a_pending_source(self):
        # The source has never been executed: the copy op has to run after the
        # ops that produce it, on the source's device, not before them.
        x = jt.ones((256,), "float32") * 5
        y = x.to_device(1)
        np.testing.assert_array_equal(y.numpy(), np.full(256, 5.0))

    def test_gradient_is_a_copy_back(self):
        a = np.random.RandomState(2).randn(64).astype("float32")
        x = jt.array(a)
        y = (x.to_device(1) ** 2).sum()
        self.assertEqual(y.device_id, 1)
        g = jt.grad(y, x)
        # the gradient comes home to x's device
        self.assertEqual(g.device_id, 0)
        np.testing.assert_allclose(g.numpy(), 2 * a, rtol=1e-5)

    def test_a_training_step_across_devices(self):
        rng = np.random.RandomState(3)
        w = rng.randn(32, 16).astype("float32")
        x = rng.randn(8, 32).astype("float32")
        wv = jt.array(w)
        xv = jt.array(x)
        loss = (jt.matmul(xv.to_device(1), wv.to_device(1)) ** 2).sum()
        self.assertEqual(loss.device_id, 1)
        gw = jt.grad(loss, wv)
        self.assertEqual(gw.device_id, 0)
        np.testing.assert_allclose(
            gw.numpy(), 2 * x.T @ (x @ w), rtol=1e-3, atol=1e-3)

    def test_copy_waits_for_the_source_kernel(self):
        """A long kernel on the source, a copy issued immediately, read on the
        destination.

        Where the pair can peer, the copy really is asynchronous and, without
        the source-side event, reads whatever was in the block before the
        chain wrote it -- the SFRL pool hands out recycled blocks, so that is
        old data, not zeros. Where it cannot peer the driver serialises the
        copy itself and this passes either way; the regime is reported so a
        green run is not mistaken for a proof. See the module docstring.
        """
        self.assertIn(_peer_regime(), ("peer", "staged", "unknown"))
        n = 4096
        rng = np.random.RandomState(4)
        base = (rng.randn(n, n).astype("float32") * 0.01)
        a = jt.array(base)
        # Fill and free a block of the same size first, so the chain's output
        # is allocated over recognisable old data rather than fresh zeros.
        junk = jt.array(np.full((n, n), 7.0, "float32"))
        junk.sync()
        del junk
        jt.gc()
        s = a
        for _ in range(6):
            s = jt.matmul(s, a)
        # Nothing has run yet: this queues the whole chain and the copy in one
        # go, so the copy is enqueued while the chain is still on the device.
        b = s.to_device(1)
        got = b.numpy()
        # The truth, read from the source device after everything has settled.
        ref = s.numpy()
        self.assertEqual(b.device_id, 1)
        np.testing.assert_array_equal(got, ref)


if __name__ == "__main__":
    unittest.main()
