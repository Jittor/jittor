# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The cuDNN RNN agrees with jittor's own recurrence, forward and backward.

``nn.RNN`` has two implementations of the same math: on CUDA it flattens its
weights into cuDNN's layout and calls ``cudnn_rnn``; everywhere else it runs
the recurrence out of jittor ops. Same module, same weights, two kernels --
which makes each an independent check on the other.

This exists because the cuDNN half is about to be rewritten onto the v8 RNN
API (8.04), and the coverage it had did not support that:

* ``tests/nn/test_rnn.py`` compares against ``import torch``, and under the
  jt-torch shim that is jittor compared against itself. All 36 of its cases
  skip here for "no torch found" anyway.
* ``tests/nn/test_rnn_recurrence.py`` does pin the recurrence against an
  independent numpy oracle -- but its *gradient* cases run on CPU only
  (it says so: "RNN uses matmul -> float32 on CPU"), so the cuDNN backward,
  which is the half with the reserve space and the weight-space layout, had
  no numeric check at all.

So: forward, both hidden states, the input gradient and every weight gradient,
across the four cell types and the layer/direction shapes that change the
weight-space layout.
"""
import unittest

import numpy as np

import jittor as jt
from jittor import nn


def _params(layer):
    """The module's own weights, in a fixed order."""
    return [(name, p) for name, p in sorted(layer.named_parameters())]


def _run(layer, x_np, h_np, c_np, use_cuda):
    """Forward + backward of `layer` on one backend; returns numpy arrays."""
    with jt.flag_scope(use_cuda=use_cuda):
        x = jt.array(x_np)
        h = jt.array(h_np)
        if c_np is None:
            out, hidden = layer(x, h)
            hiddens = [hidden]
        else:
            c = jt.array(c_np)
            out, (hn, cn) = layer(x, (h, c))
            hiddens = [hn, cn]
        # A fixed, non-uniform cotangent: summing the output would hide any
        # error that cancels across the sequence.
        seed = jt.array(
            np.linspace(-1.0, 1.0, int(np.prod(out.shape)))
            .reshape(out.shape).astype("float32"))
        loss = (out * seed).sum()
        names = [n for n, _ in _params(layer)]
        grads = jt.grad(loss, [x] + [p for _, p in _params(layer)])
        wanted = [out] + hiddens + grads
        got = jt.fetch_sync(list(wanted))
    return names, got


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudnnRnnMatchesTheRecurrence(unittest.TestCase):
    SEQ, BATCH, INPUT, HIDDEN = 5, 3, 4, 6

    def _check(self, cls, mode, **kw):
        rng = np.random.RandomState(0)
        num_layers = kw.get("num_layers", 1)
        dirs = 2 if kw.get("bidirectional", False) else 1
        layer = cls(self.INPUT, self.HIDDEN, **kw)
        x = rng.randn(self.SEQ, self.BATCH, self.INPUT).astype("float32")
        h = rng.randn(num_layers * dirs, self.BATCH, self.HIDDEN).astype("float32")
        c = (rng.randn(num_layers * dirs, self.BATCH, self.HIDDEN).astype("float32")
             if mode == "lstm" else None)

        names, want = _run(layer, x, h, c, use_cuda=0)
        _, got = _run(layer, x, h, c, use_cuda=1)

        labels = ["output"] + (["h_n", "c_n"] if mode == "lstm" else ["h_n"])
        labels += ["d_input"] + ["d_" + n for n in names]
        self.assertEqual(len(labels), len(want))
        for label, a, b in zip(labels, got, want):
            scale = max(1.0, float(np.abs(b).max()))
            err = float(np.abs(a - b).max()) / scale
            self.assertLess(
                err, 5e-6,
                "%s mismatch %.3g between cuDNN and the recurrence (%s %s)"
                % (label, err, mode, kw))

    def test_rnn_tanh(self):
        self._check(nn.RNN, "rnn", nonlinearity="tanh")

    def test_rnn_relu(self):
        self._check(nn.RNN, "rnn", nonlinearity="relu")

    def test_lstm(self):
        self._check(nn.LSTM, "lstm")

    def test_gru(self):
        self._check(nn.GRU, "gru")

    # ---- the shapes that change the weight-space layout -----------------
    def test_lstm_multilayer(self):
        self._check(nn.LSTM, "lstm", num_layers=2)

    def test_lstm_bidirectional(self):
        self._check(nn.LSTM, "lstm", bidirectional=True)

    def test_gru_multilayer_bidirectional(self):
        self._check(nn.GRU, "gru", num_layers=2, bidirectional=True)

    def test_rnn_no_bias(self):
        """Bias off changes which offsets the flat weight uses."""
        self._check(nn.RNN, "rnn", nonlinearity="tanh", bias=False)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestRnnFollowsTheFloat32Policy(unittest.TestCase):
    """`float32_matmul_precision` reaches the RNN, and it is measurable.

    The RNN set no math type for float32, so it got cuDNN's default -- which
    on Ampere and later *allows tf32*. An fp32 LSTM ran at tf32 whatever
    ``cuda_allow_cudnn_tf32`` said, and nothing could turn it off.

    Unlike the convolution side, where permitting tensor-op numerics only
    makes those engines eligible and cuDNN may still pick FMA, here the
    permission changes the answer by three orders of magnitude. So this is
    asserted on the numbers rather than on a log line: the tier has to be
    *worse* at ``medium`` than at ``highest``, which is the only way to tell
    "tf32 was allowed" from "tf32 was never asked about".
    """

    def setUp(self):
        self._saved = (jt.flags.float32_matmul_precision,
                       jt.flags.cuda_allow_cudnn_tf32)
        jt.flags.cuda_allow_cudnn_tf32 = 0

    def tearDown(self):
        jt.sync_all()
        (jt.flags.float32_matmul_precision,
         jt.flags.cuda_allow_cudnn_tf32) = self._saved

    def _error_against_float64(self, tier):
        rng = np.random.RandomState(0)
        seq, batch, isize, hidden = 5, 3, 4, 6
        layer = nn.LSTM(isize, hidden)
        x = rng.randn(seq, batch, isize).astype("float32")
        h = rng.randn(1, batch, hidden).astype("float32")
        c = rng.randn(1, batch, hidden).astype("float32")

        # float64 reference, on the CPU recurrence.
        saved = [(n, p.numpy()) for n, p in _params(layer)]
        with jt.flag_scope(use_cuda=0):
            for _, p in _params(layer):
                p.assign(p.float64())
            xr, hr, cr = (jt.array(v.astype("float64")) for v in (x, h, c))
            out, _ = layer(xr, (hr, cr))
            seed = jt.array(np.linspace(-1.0, 1.0, int(np.prod(out.shape)))
                            .reshape(out.shape).astype("float64"))
            gref = jt.grad((out * seed).sum(),
                           [p for _, p in _params(layer)])
            ref = jt.fetch_sync(list(gref))
        for (name, value), (_, p) in zip(saved, _params(layer)):
            p.assign(jt.array(value))

        jt.flags.float32_matmul_precision = tier
        _, got = _run(layer, x, h, c, use_cuda=1)
        # skip output/h_n/c_n/d_input; compare the weight gradients
        got_grads = got[4:]
        return max(float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max()))
                   for a, b in zip(got_grads, ref))

    def test_highest_is_true_float32_and_medium_is_not(self):
        highest = self._error_against_float64("highest")
        medium = self._error_against_float64("medium")
        # Measured on sm_89 / cuDNN 8.9.7: highest 1.1e-07, medium 2.3e-04.
        self.assertLess(highest, 1e-6,
                        "highest is not true float32: %.3g" % highest)
        self.assertGreater(
            medium, highest * 20,
            "medium (%.3g) is no worse than highest (%.3g): the precision "
            "policy is not reaching the RNN descriptor" % (medium, highest))


if __name__ == "__main__":
    unittest.main()
