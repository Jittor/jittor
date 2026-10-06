"""Channels-last activations: NHWC storage read as NCHW, kept through a model.

A half-precision convolution that records no gradient answers with an NCHW
view of NHWC storage; elementwise operators, group norm and max pooling keep
that layout, so the next convolution reads NHWC memory as it is. In training a
convolution offers the channels-last result and a batch norm takes it
(`channels_last_training`); gradients come back the same way. Values never
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


class _ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

    def execute(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.max_pool2d(x, 3, 2, 1)
        x = nn.relu(self.bn2(self.conv2(x)) + x)
        x = nn.relu(self.bn3(self.conv3(x)))
        return x.mean(dims=(2, 3))


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

    def test_a_gradient_graph_keeps_the_layout(self):
        # Forward and back: the gradient reaching a channels-last view is a
        # view of the same layout, not a dense copy.
        a = self.rng.randn(2, 8, 5, 6).astype("float32")
        x = _nhwc_view(a).float32()
        x.start_grad()
        y = x * 2.0
        self.assertTrue(_is_channels_last(y))
        g = jt.grad((y * y).sum(), x)
        np.testing.assert_allclose(g.numpy(), 8.0 * a, rtol=1e-2, atol=1e-2)

    def _training_net(self):
        # Parameters from NumPy: the device generators differ, so the CPU run
        # would not otherwise start from the same weights.
        net = _ResidualNet()
        rng = np.random.RandomState(1)
        for p in net.parameters():
            if not p.is_stop_grad():
                p.assign(jt.array((rng.randn(*p.shape) * 0.2).astype("float32")))
        return net, [p for p in net.parameters() if not p.is_stop_grad()]

    def _train_step(self, channels_last):
        before = cudnn_backend.channels_last_training
        cudnn_backend.channels_last_training = channels_last
        try:
            net, params = self._training_net()
            x = jt.array(self.rng_train)
            loss = (net(x) ** 2).mean()
            grads = jt.grad(loss, params)
            stats = [net.bn1.running_mean, net.bn1.running_var, net.bn3.running_mean]
            return (loss.numpy(), [g.numpy() for g in grads], [s.numpy() for s in stats])
        finally:
            cudnn_backend.channels_last_training = before

    def test_a_training_step_matches_nchw(self):
        # Convolution, batch norm with relu and with a residual add, max
        # pooling, a strided convolution and average pooling, in float32:
        # channels-last on CUDA against the NCHW kernels on CUDA and against
        # the same graph on the CPU.
        self.rng_train = self.rng.randn(4, 3, 20, 20).astype("float32")
        with jt.flag_scope(auto_graph_replay=0, use_tensorcore=0):
            got = self._train_step(True)
            nchw = self._train_step(False)
            with jt.flag_scope(use_cuda=0):
                cpu = self._train_step(False)
        for want, rtol in ((nchw, 1e-4), (cpu, 1e-3)):
            np.testing.assert_allclose(got[0], want[0], rtol=rtol)
            # The scale of all the gradients: the convolution bias ahead of a
            # batch norm has a gradient of zero up to rounding.
            scale = max(np.abs(w).max() for w in want[1])
            for g, w in zip(got[1], want[1]):
                np.testing.assert_allclose(g, w, rtol=rtol, atol=10 * rtol * 1e-2 * scale)
            for g, w in zip(got[2], want[2]):
                np.testing.assert_allclose(g, w, rtol=rtol, atol=1e-5)

    def test_a_training_convolution_goes_channels_last_only_into_a_batch_norm(self):
        # A group norm's convolution output is read by the residual add as
        # well, which would compute it twice; it stays NCHW.
        conv = nn.Conv2d(3, 8, 3, padding=1)
        bn = nn.BatchNorm2d(8)
        gn = nn.GroupNorm(2, 8)
        x = jt.array(self.rng.randn(2, 3, 6, 6).astype("float32"))
        self.assertTrue(conv(x)._storage_is_contiguous())
        self.assertTrue(_is_channels_last(bn(conv(x))))
        self.assertTrue(gn(conv(x))._storage_is_contiguous())

    def test_a_training_group_norm_over_channels_last_storage(self):
        # Ten channels a group, so a float4 of channels straddles two groups;
        # with silu taken into the pass, and its gradient into the backward's.
        for shape, groups, act in (((2, 320, 6, 5), 32, True), ((3, 8, 7, 4), 2, False)):
            a = (self.rng.randn(*shape) * 2 + 0.5).astype("float32")
            w = (self.rng.randn(shape[1]) * 0.5 + 1).astype("float32")
            b = self.rng.randn(shape[1]).astype("float32")
            cot = self.rng.randn(*shape).astype("float32")

            def run(channels_last, use_cuda):
                with jt.flag_scope(use_cuda=use_cuda):
                    if channels_last:
                        source = jt.array(np.ascontiguousarray(a.transpose(0, 2, 3, 1)))
                        x = source._storage_permute((0, 3, 1, 2))
                    else:
                        source = x = jt.array(a)
                    jw, jb = jt.array(w), jt.array(b)
                    y = nn.group_norm(x, groups, jw, jb, 1e-5)
                    if act:
                        y = nn.silu(y)
                    if channels_last:
                        self.assertIsNotNone(channels_last_source(y))
                    gs, gw, gb = jt.grad((y * jt.array(cot)).sum(), [source, jw, jb])
                    gs = gs.numpy()
                    if channels_last:
                        gs = gs.transpose(0, 3, 1, 2)
                    return [y.numpy(), gs, gw.numpy(), gb.numpy()]
            got = run(True, 1)
            want = run(False, 0)
            for name, g, r in zip(("y", "dx", "dw", "db"), got, want):
                np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-4 * np.abs(r).max(),
                                           err_msg=f"{shape} {name}")

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

    def test_a_large_group_norm_finishes_its_rows_in_the_last_segment(self):
        # Many segments a row, so the row's last segment to store its partial
        # is the one that finishes it; called again, as a replay would, the
        # same answer -- the per-row count went back to zero.
        a = (self.rng.randn(2, 320, 32, 32) * 2 + 0.5).astype("float32")
        weight, bias = self.rng.randn(320).astype("float32"), self.rng.randn(320).astype("float32")
        x = _nhwc_view(a)
        ref = x.float64().numpy().reshape(2, 32, -1)
        mean, var = ref.mean(-1, keepdims=True), ref.var(-1, keepdims=True)
        want = ((ref - mean) / np.sqrt(var + 1e-5)).reshape(a.shape) \
            * weight[None, :, None, None] + bias[None, :, None, None]
        with jt.no_grad():
            for _ in range(3):
                got = nn.group_norm(x, 32, jt.array(weight).float16(), jt.array(bias).float16(), 1e-5)
                self.assertIsNotNone(channels_last_source(got))
                np.testing.assert_allclose(got.float32().numpy(), want, rtol=2e-2, atol=2e-2)

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

    def test_a_convolution_reads_through_a_pending_contiguous_copy(self):
        # Diffusers' `conv_shortcut(input.contiguous())`: the convolution reads
        # the channels-last storage and the dense copy is never made.
        conv1 = nn.Conv2d(8, 16, 3, padding=1)
        conv2 = nn.Conv2d(16, 4, 1)
        for m in (conv1, conv2):
            for p in m.parameters():
                p.assign(p.float16())
        a = self.rng.randn(1, 8, 12, 12).astype("float32")
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0):
            x = jt.array(a).float16()
            x.sync()
            before = cudnn_backend.channels_last_activations
            try:
                cudnn_backend.channels_last_activations = False
                want = conv2(conv1(x).contiguous()).float32().numpy()
                cudnn_backend.channels_last_activations = True
                h = conv1(x)
                self.assertTrue(_is_channels_last(h))
                h.sync()
                jt.sync_all(True)
                with jt.profile() as p:
                    got = conv2(h.contiguous())
                    got.sync()
                    jt.sync_all(True)
            finally:
                cudnn_backend.channels_last_activations = before
        names = [dict(k)["name"] if not isinstance(k, dict) else k["name"]
                 for k in p.result.kernel_records]
        self.assertFalse(any("contiguous" in n or "transpose" in n.lower() for n in names), names)
        np.testing.assert_allclose(got.float32().numpy(), want, rtol=2e-2, atol=2e-2)

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
