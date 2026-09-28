"""Channels-last activations: NHWC storage read as NCHW, kept through a model.

A half-precision convolution that records no gradient answers with an NCHW
view of NHWC storage; elementwise operators, group norm and max pooling keep
that layout, so the next convolution reads NHWC memory as it is. Values never
depend on it: every check here compares against the dense NCHW computation.

Run::  python -m pytest tests/nn/test_channels_last.py
"""
from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt
from jittor import nn
import jittor.nn.backends.cudnn as cudnn_backend
from jittor.nn.functional._layout import channels_last_source

HAS_CUDA = _test_capability.check_accelerator("cuda", backend=jt).enabled


def _nhwc_view(a):
    """A float16 NCHW view of dense NHWC storage holding `a`."""
    source = jt.array(np.ascontiguousarray(a.transpose(0, 2, 3, 1))).float16()
    source.sync()
    return source._storage_permute((0, 3, 1, 2))


def _is_channels_last(v):
    n, c, h, w = v.shape
    return tuple(v._storage_strides()) == (h * w * c, 1, w * c, c)


@unittest.skipIf(not HAS_CUDA, "needs CUDA")
class TestChannelsLast(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1)
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)
        self.rng = np.random.RandomState(0)

    def test_elementwise_keeps_the_layout(self):
        a = self.rng.randn(2, 8, 5, 6).astype("float32")
        scale = self.rng.randn(8).astype("float32")
        with jt.no_grad():
            x = _nhwc_view(a)
            s = jt.array(scale).float16()
            y = nn.relu(x * s.broadcast(x, [0, 2, 3]) + 1.0)
            z = jt.ternary(y > 0.5, y, x)
            self.assertTrue(_is_channels_last(y))
            self.assertTrue(_is_channels_last(z))
            want_y = np.maximum(a * scale.reshape(1, 8, 1, 1) + 1.0, 0)
            np.testing.assert_allclose(y.float32().numpy(), want_y, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(z.float32().numpy(),
                                       np.where(want_y > 0.5, want_y, a), rtol=1e-2, atol=1e-2)
            # An operand already in memory (here: dense NCHW) is read through
            # a view; the answer is the same.
            dense = jt.array(a).float16()
            dense.sync()
            both = x + dense
            np.testing.assert_allclose(both.float32().numpy(), 2 * a, rtol=1e-2, atol=1e-2)

    def test_a_gradient_graph_is_left_dense(self):
        a = self.rng.randn(2, 8, 5, 6).astype("float32")
        x = _nhwc_view(a).float32()
        x.start_grad()
        y = x * 2.0
        self.assertTrue(y._storage_is_contiguous())
        g = jt.grad(y.sum(), x)
        np.testing.assert_allclose(g.numpy(), np.full_like(a, 2.0))

    def test_convolution_group_norm_and_pooling_match_nchw(self):
        conv1 = nn.Conv2d(16, 32, 3, padding=1)
        conv2 = nn.Conv2d(32, 32, 3, padding=1)
        norm = nn.GroupNorm(8, 32)
        for m in (conv1, conv2, norm):
            for p in m.parameters():
                p.assign(p.float16())

        def run():
            x = jt.array(self.rng_input).float16()
            h = nn.silu(norm(conv1(x)))
            h = nn.max_pool2d(h, 3, 2, 1)
            return conv2(h), h
        self.rng_input = self.rng.randn(2, 16, 12, 12).astype("float32")
        # Replay would serve the second call from the first one's recording.
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0):
            before = cudnn_backend.channels_last_activations
            try:
                cudnn_backend.channels_last_activations = False
                want, want_mid = (t.float32().numpy() for t in run())
                cudnn_backend.channels_last_activations = True
                got, mid = run()
                self.assertTrue(_is_channels_last(mid))
                self.assertIsNotNone(channels_last_source(got))
                got = got.float32().numpy()
            finally:
                cudnn_backend.channels_last_activations = before
        np.testing.assert_allclose(mid.float32().numpy(), want_mid, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())

    def test_interpolation_keeps_the_layout(self):
        a = self.rng.randn(2, 8, 5, 6).astype("float32")
        with jt.no_grad():
            dense = jt.array(a).float16()
            view = _nhwc_view(a)
            for mode, size in (("nearest", (10, 12)), ("bilinear", (7, 9)),
                               ("bicubic", (10, 12))):
                with self.subTest(mode=mode):
                    got = nn.interpolate(view, size=size, mode=mode)
                    want = nn.interpolate(dense, size=size, mode=mode)
                    self.assertTrue(_is_channels_last(got))
                    np.testing.assert_allclose(got.float32().numpy(), want.float32().numpy(),
                                               rtol=1e-2, atol=1e-2)

    def test_copy_from_a_strided_source(self):
        a = self.rng.randn(2, 4, 3, 5).astype("float32")
        with jt.no_grad():
            v = _nhwc_view(a)
            out = jt.empty(v.shape, "float16")
            out.sync()
            out._copy_into(v)
        np.testing.assert_allclose(out.float32().numpy(), a, rtol=1e-2, atol=1e-2)

    def test_a_view_of_a_view_composes(self):
        a = self.rng.randn(2, 4, 3, 5).astype("float32")
        v = _nhwc_view(a)
        back = v._storage_permute((0, 2, 3, 1))
        self.assertTrue(back._storage_is_contiguous())
        np.testing.assert_allclose(back.float32().numpy(), a.transpose(0, 2, 3, 1),
                                   rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
