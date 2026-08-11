# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Recurrent layers (RNN / LSTM / GRU) -- forward recurrence + backward correctness vs an
INDEPENDENT numpy oracle.

The legacy ``test_rnn.py`` compares against ``import torch`` -- which under the jt-torch
shim is jittor comparing against itself (a tautological oracle that always "passes"). This
module instead pins the recurrence against a from-scratch numpy implementation, so a green
result means jittor matches the textbook math, not itself.

The recurrence is the bug-prone part: a wrong gate order, a dropped ``b_hh`` bias, or a
backward that does not unroll through time produces plausible output yet wrong training
gradients. Both halves are checked:

  * **Forward** -- the layer's own weights are extracted and fed to a numpy recurrence
    (PyTorch equations: RNN tanh/relu; LSTM gates i,f,g,o; GRU gates r,z,n). Output must
    match, on CPU and the accelerator (the cuDNN RNN path is a separate kernel).
  * **Backward** -- jittor's analytic ``d sum(output)/d input`` vs a float64 central finite
    difference of the numpy reference (fully independent of jittor autograd).

Covers single-layer, multi-layer and bidirectional configs.

Run::  python -m pytest tests/nn/test_rnn_recurrence.py
"""
import unittest

import numpy as np
import jittor as jt
from jittor import nn

from _helpers.common import (
    JittorTestCase, get_all_device_types, use_cuda_for, to_numpy,
)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------- numpy reference cells
def _rnn_layer(x, w_ih, w_hh, b_ih, b_hh, h0, nonlin):
    f = np.tanh if nonlin == "tanh" else (lambda z: np.maximum(z, 0.0))
    h = h0
    outs = []
    for t in range(x.shape[0]):
        h = f(x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh)
        outs.append(h)
    return np.stack(outs), h


def _lstm_layer(x, w_ih, w_hh, b_ih, b_hh, h0, c0):
    H = w_hh.shape[1]
    h, c = h0, c0
    outs = []
    for t in range(x.shape[0]):
        g = x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh        # (B, 4H)
        i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
        i, f, gg, o = _sig(i), _sig(f), np.tanh(gg), _sig(o)
        c = f * c + i * gg
        h = o * np.tanh(c)
        outs.append(h)
    return np.stack(outs), h, c


def _gru_layer(x, w_ih, w_hh, b_ih, b_hh, h0):
    H = w_hh.shape[1]
    h = h0
    outs = []
    for t in range(x.shape[0]):
        gi = x[t] @ w_ih.T + b_ih                           # (B, 3H)
        gh = h @ w_hh.T + b_hh
        ir, iz, in_ = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        hr, hz, hn = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]
        r = _sig(ir + hr)
        z = _sig(iz + hz)
        n = np.tanh(in_ + r * hn)
        h = (1.0 - z) * n + z * h
        outs.append(h)
    return np.stack(outs), h


def _params(mod, layer, suffix=""):
    g = lambda n: to_numpy(getattr(mod, n)).astype("float64")
    return (g(f"weight_ih_l{layer}{suffix}"), g(f"weight_hh_l{layer}{suffix}"),
            g(f"bias_ih_l{layer}{suffix}"), g(f"bias_hh_l{layer}{suffix}"))


def _ref_forward(mod, kind, x, num_layers, bidirectional, nonlin="tanh"):
    """numpy reference matching nn.{RNN,LSTM,GRU} over layers/directions ('output')."""
    B = x.shape[1]
    inp = x
    for layer in range(num_layers):
        H = mod.hidden_size
        dirs = []
        suffixes = ["", "_reverse"] if bidirectional else [""]
        for d, suffix in enumerate(suffixes):
            w_ih, w_hh, b_ih, b_hh = _params(mod, layer, suffix)
            xi = inp[::-1] if d == 1 else inp
            h0 = np.zeros((B, H))
            if kind == "rnn":
                out, _ = _rnn_layer(xi, w_ih, w_hh, b_ih, b_hh, h0, nonlin)
            elif kind == "lstm":
                out, _, _ = _lstm_layer(xi, w_ih, w_hh, b_ih, b_hh, h0, np.zeros((B, H)))
            else:
                out, _ = _gru_layer(xi, w_ih, w_hh, b_ih, b_hh, h0)
            dirs.append(out[::-1] if d == 1 else out)
        inp = np.concatenate(dirs, axis=2) if bidirectional else dirs[0]
    return inp


class TestRNNForward(JittorTestCase):
    """Layer output == from-scratch numpy recurrence, on every device."""

    def _devices(self, body):
        for dev in get_all_device_types():
            with self.subTest(device=dev):
                with jt.flag_scope(use_cuda=use_cuda_for(dev)):
                    body(dev)

    def _make(self, ctor, kind, num_layers=1, bidirectional=False, **kw):
        x0 = np.random.RandomState(0).randn(4, 3, 5).astype("float32")  # (T, B, I)
        mod = ctor(5, 6, num_layers=num_layers, bidirectional=bidirectional, **kw)

        def body(dev):
            x = jt.array(x0)
            out = mod(x)
            out = out[0] if isinstance(out, tuple) else out
            ref = _ref_forward(mod, kind, x0.astype("float64"),
                               num_layers, bidirectional, kw.get("nonlinearity", "tanh"))
            self.assertEqual(to_numpy(out), ref, atol=1e-4, rtol=1e-4,
                             msg=f"{kind} forward vs numpy recurrence [{dev}]")
        self._devices(body)

    def test_rnn_tanh(self):           self._make(nn.RNN, "rnn", nonlinearity="tanh")
    def test_rnn_relu(self):           self._make(nn.RNN, "rnn", nonlinearity="relu")
    def test_lstm(self):               self._make(nn.LSTM, "lstm")
    def test_gru(self):                self._make(nn.GRU, "gru")
    def test_rnn_multilayer(self):     self._make(nn.RNN, "rnn", num_layers=2)
    def test_lstm_multilayer(self):    self._make(nn.LSTM, "lstm", num_layers=2)
    def test_rnn_bidirectional(self):  self._make(nn.RNN, "rnn", bidirectional=True)
    def test_lstm_bidirectional(self): self._make(nn.LSTM, "lstm", bidirectional=True)


class TestRNNBackward(JittorTestCase):
    """jittor analytic d sum(out)/d input == numpy-reference finite difference (float64)."""

    def _fd_input_grad(self, ref_fn, x0):
        eps = 1e-5
        g = np.zeros_like(x0, dtype="float64")
        flat = x0.astype("float64").reshape(-1)
        for k in range(flat.size):
            xp = flat.copy(); xp[k] += eps
            xm = flat.copy(); xm[k] -= eps
            g.reshape(-1)[k] = (ref_fn(xp.reshape(x0.shape)).sum()
                                - ref_fn(xm.reshape(x0.shape)).sum()) / (2 * eps)
        return g

    def _check(self, ctor, kind, **kw):
        x0 = np.random.RandomState(1).randn(3, 2, 4).astype("float32")  # small for FD
        mod = ctor(4, 5, num_layers=1, **kw)
        with jt.flag_scope(use_cuda=0):                # RNN uses matmul -> float32 on CPU
            x = jt.array(x0)
            out = mod(x)
            out = out[0] if isinstance(out, tuple) else out
            gj = to_numpy(jt.grad(out.sum(), x))
        ref_fn = lambda xx: _ref_forward(mod, kind, xx.astype("float64"), 1, False,
                                         kw.get("nonlinearity", "tanh"))
        gfd = self._fd_input_grad(ref_fn, x0)
        self.assertEqual(gj.astype("float64"), gfd, atol=2e-2, rtol=2e-2,
                         msg=f"{kind} backward (analytic vs numpy-ref FD)")

    def test_rnn_tanh_grad(self):  self._check(nn.RNN, "rnn", nonlinearity="tanh")
    def test_lstm_grad(self):      self._check(nn.LSTM, "lstm")
    def test_gru_grad(self):       self._check(nn.GRU, "gru")


if __name__ == "__main__":
    unittest.main(verbosity=2)
