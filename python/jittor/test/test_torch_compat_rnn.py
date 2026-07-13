"""Torch-grade RNN/LSTM/GRU parity for ``import jittor as torch``.

Forward shapes (incl. batch_first), hidden-state shapes, finite backward, and a 1-step
LSTM-cell check against an explicit numpy reference. CPU+CUDA.

Run:  python -m jittor.test.test_torch_compat_rnn
"""
import unittest
import numpy as np
import jittor as jt
from jittor import nn

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def shape(self, v, exp, msg=""):
        self.assertEqual(tuple(v.shape), tuple(exp), f"{msg}: {tuple(v.shape)}!={tuple(exp)}")


class TestRNNShapes(Base):
    def test_lstm_shapes(self):
        L, B, I, H = 5, 2, 4, 6
        def body(dev):
            x = jt.randn(L, B, I)
            m = nn.LSTM(I, H, num_layers=2)
            out, (h, c) = m(x)
            self.shape(out, (L, B, H), f"lstm out {dev}")
            self.shape(h, (2, B, H), f"lstm h {dev}")
            self.shape(c, (2, B, H), f"lstm c {dev}")
        both_devices(body)

    def test_gru_shapes(self):
        L, B, I, H = 4, 3, 5, 7
        def body(dev):
            x = jt.randn(L, B, I)
            m = nn.GRU(I, H)
            out, h = m(x)
            self.shape(out, (L, B, H), f"gru out {dev}")
            self.shape(h, (1, B, H), f"gru h {dev}")
        both_devices(body)

    def test_rnn_batch_first(self):
        B, L, I, H = 2, 5, 4, 6
        def body(dev):
            x = jt.randn(B, L, I)
            m = nn.RNN(I, H, batch_first=True)
            out, h = m(x)
            self.shape(out, (B, L, H), f"rnn batch_first out {dev}")
        both_devices(body)

    def test_lstm_backward_finite(self):
        def body(dev):
            x = jt.randn(4, 2, 3)
            m = nn.LSTM(3, 5)
            out, _ = m(x)
            g = jt.grad(out.sum(), [p for p in m.parameters() if not p.is_stop_grad()])
            self.assertTrue(all(bool(jt.isfinite(gi).all().item()) for gi in g),
                            f"lstm grads finite {dev}")
        both_devices(body)


class TestLSTMCell(Base):
    def test_lstm_cell_vs_reference(self):
        # one LSTM step (seq len 1) vs an explicit numpy LSTM cell, deterministic weights.
        I, H = 3, 4
        rng = np.random.RandomState(0)
        wih = rng.randn(4 * H, I).astype("float32") * 0.1
        whh = rng.randn(4 * H, H).astype("float32") * 0.1
        bih = rng.randn(4 * H).astype("float32") * 0.1
        bhh = rng.randn(4 * H).astype("float32") * 0.1
        x = rng.randn(1, 1, I).astype("float32")

        def sig(z): return 1.0 / (1.0 + np.exp(-z))

        def ref():
            xt = x[0, 0]
            gates = wih @ xt + bih + whh @ np.zeros(H, "float32") + bhh
            i, f, g, o = gates[:H], gates[H:2*H], gates[2*H:3*H], gates[3*H:]
            i, f, o = sig(i), sig(f), sig(o)
            g = np.tanh(g)
            c = f * 0 + i * g
            h = o * np.tanh(c)
            return h

        def body(dev):
            m = nn.LSTM(I, H)
            m.weight_ih_l0 = jt.array(wih); m.weight_hh_l0 = jt.array(whh)
            m.bias_ih_l0 = jt.array(bih); m.bias_hh_l0 = jt.array(bhh)
            out, _ = m(jt.array(x))
            np.testing.assert_allclose(np.asarray(out.numpy())[0, 0], ref(),
                                       atol=1e-4, rtol=1e-4, err_msg=f"lstm cell {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
