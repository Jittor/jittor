"""Torch-grade nn/functional-semantics regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (round 3). Like ``test_torch_compat_ops.py``
this is a structured ``unittest`` module: every check compares jittor-as-torch's
``nn.functional`` / ``nn`` against an INDEPENDENT, explicit numpy reference, and runs on
BOTH CPU and CUDA (when the build has it), so it locks torch *nn semantics* rather than
jittor self-consistency.

Covered functional ops: ``relu`` / ``gelu`` (exact erf form, torch default) / ``silu`` /
``sigmoid`` / ``tanh`` / ``softmax`` / ``log_softmax`` / ``leaky_relu`` / ``elu`` /
``relu6``, ``linear``; losses ``cross_entropy`` / ``nll_loss`` / ``mse_loss`` /
``l1_loss`` / ``binary_cross_entropy`` (+ ``_with_logits``) / ``kl_div`` (mean /
batchmean / sum); norms ``layer_norm`` / ``group_norm``. Modules: ``nn.Linear`` /
``nn.ReLU`` / ``nn.LayerNorm`` forward.

Notes:
  * ``F.tanh`` / ``F.sigmoid`` are not both on jittor's ``nn.functional``; ``tanh`` is the
    elementwise ``torch.tanh`` (tested via the module-level op). ``F.sigmoid`` exists.
  * jittor has no 0-d scalars; reduced losses are shape ``(1,)`` -> compared via
    ``.item()`` against the numpy Python scalar.
  * ``kl_div`` default reduction is torch's ``'mean'`` = elementwise-sum / NUMEL (NOT
    batchmean) -- the reference matches that.

Run:  python -m jittor.test.test_torch_compat_nn
      python -m pytest python/jittor/test/test_torch_compat_nn.py
"""
import math
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
from jittor import nn

F = nn.functional

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def t(a):
    return torch.array(a)


_verf = np.vectorize(math.erf)


def np_softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def scalar(self, got, ref, places=4, msg=""):
        # jittor reduced loss is shape (1,); compare the python value.
        self.assertAlmostEqual(float(np.asarray(got).reshape(-1)[0]), float(ref),
                               places=places, msg=msg)


# ------------------------------------------------------------------------ activations

class TestActivations(Base):
    def setUp(self):
        self.x = np.random.RandomState(0).randn(4, 5).astype("float32")

    def test_relu(self):
        x = self.x
        def body(dev):
            self.ac(F.relu(t(x)).numpy(), np.maximum(x, 0), msg=f"relu {dev}")
        both_devices(body)

    def test_relu6(self):
        x = self.x * 5
        def body(dev):
            self.ac(F.relu6(t(x)).numpy(), np.minimum(np.maximum(x, 0), 6),
                    msg=f"relu6 {dev}")
        both_devices(body)

    def test_sigmoid(self):
        x = self.x
        def body(dev):
            self.ac(F.sigmoid(t(x)).numpy(), 1.0 / (1.0 + np.exp(-x)), atol=1e-6,
                    msg=f"sigmoid {dev}")
        both_devices(body)

    def test_tanh(self):
        # tanh is the elementwise op torch.tanh (not on jittor's nn.functional).
        x = self.x
        def body(dev):
            self.ac(torch.tanh(t(x)).numpy(), np.tanh(x), atol=1e-6, msg=f"tanh {dev}")
        both_devices(body)

    def test_silu(self):
        x = self.x
        def body(dev):
            self.ac(F.silu(t(x)).numpy(), x / (1.0 + np.exp(-x)), atol=1e-6,
                    msg=f"silu {dev}")
        both_devices(body)

    def test_gelu_exact_erf(self):
        # torch's default gelu is the exact (erf) form -- verify jittor matches that,
        # NOT the tanh approximation.
        x = self.x
        ref = 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)))
        def body(dev):
            self.ac(F.gelu(t(x)).numpy(), ref, atol=1e-5, msg=f"gelu erf {dev}")
        both_devices(body)

    def test_leaky_relu(self):
        x = self.x
        for slope in [0.01, 0.2]:
            def body(dev, slope=slope):
                self.ac(F.leaky_relu(t(x), slope).numpy(),
                        np.where(x > 0, x, slope * x), atol=1e-6,
                        msg=f"leaky_relu {slope} {dev}")
            both_devices(body)

    def test_elu(self):
        x = self.x
        ref = np.where(x > 0, x, np.exp(x) - 1.0)
        def body(dev):
            self.ac(F.elu(t(x)).numpy(), ref, atol=1e-6, msg=f"elu {dev}")
        both_devices(body)

    def test_softmax(self):
        x = self.x
        for dim in [-1, 0, 1]:
            def body(dev, dim=dim):
                self.ac(F.softmax(t(x), dim=dim).numpy(), np_softmax(x, dim), atol=1e-6,
                        msg=f"softmax dim={dim} {dev}")
            both_devices(body)

    def test_log_softmax(self):
        x = self.x
        for dim in [-1, 0]:
            def body(dev, dim=dim):
                self.ac(F.log_softmax(t(x), dim=dim).numpy(), np.log(np_softmax(x, dim)),
                        atol=1e-5, msg=f"log_softmax dim={dim} {dev}")
            both_devices(body)


# ----------------------------------------------------------------------------- linear

class TestLinear(Base):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.x = rng.randn(4, 5).astype("float32")
        self.w = rng.randn(3, 5).astype("float32")
        self.b = rng.randn(3).astype("float32")

    def test_linear_with_bias(self):
        x, w, b = self.x, self.w, self.b
        def body(dev):
            self.ac(F.linear(t(x), t(w), t(b)).numpy(), x @ w.T + b, atol=1e-4,
                    msg=f"linear+bias {dev}")
        both_devices(body)

    def test_linear_no_bias(self):
        x, w = self.x, self.w
        def body(dev):
            self.ac(F.linear(t(x), t(w)).numpy(), x @ w.T, atol=1e-4,
                    msg=f"linear no-bias {dev}")
        both_devices(body)


# ----------------------------------------------------------------------------- losses

class TestLosses(Base):
    def test_mse_loss(self):
        rng = np.random.RandomState(2)
        a = rng.randn(3, 4).astype("float32"); b = rng.randn(3, 4).astype("float32")
        def body(dev):
            self.scalar(F.mse_loss(t(a), t(b)).numpy(), ((a - b) ** 2).mean(),
                        msg=f"mse mean {dev}")
            self.scalar(F.mse_loss(t(a), t(b), reduction="sum").numpy(),
                        ((a - b) ** 2).sum(), places=2, msg=f"mse sum {dev}")
        both_devices(body)

    def test_mse_loss_none(self):
        # fixed: nn.mse_loss now special-cases reduction='none' (was forwarded to
        # Var.reduce -> "no such reduce" crash)
        rng = np.random.RandomState(20)
        a = rng.randn(3, 4).astype("float32"); b = rng.randn(3, 4).astype("float32")
        def body(dev):
            self.ac(F.mse_loss(t(a), t(b), reduction="none").numpy(), (a - b) ** 2,
                    atol=1e-5, msg=f"mse none {dev}")
        both_devices(body)

    def test_l1_loss(self):
        rng = np.random.RandomState(3)
        a = rng.randn(3, 4).astype("float32"); b = rng.randn(3, 4).astype("float32")
        def body(dev):
            self.scalar(F.l1_loss(t(a), t(b)).numpy(), np.abs(a - b).mean(),
                        msg=f"l1 mean {dev}")
        both_devices(body)

    def test_l1_loss_reduction(self):
        # fixed: nn.l1_loss now takes reduction= (none/sum/mean); was hard-coded mean
        rng = np.random.RandomState(30)
        a = rng.randn(3, 4).astype("float32"); b = rng.randn(3, 4).astype("float32")
        def body(dev):
            self.ac(F.l1_loss(t(a), t(b), reduction="none").numpy(), np.abs(a - b),
                    atol=1e-5, msg=f"l1 none {dev}")
        both_devices(body)

    def test_cross_entropy(self):
        rng = np.random.RandomState(4)
        logits = rng.randn(5, 3).astype("float32")
        tgt = np.array([0, 1, 2, 1, 0], dtype="int64")
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        ref = -logp[np.arange(5), tgt].mean()
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt)).numpy(), ref,
                        msg=f"cross_entropy {dev}")
        both_devices(body)

    def test_cross_entropy_sum(self):
        rng = np.random.RandomState(40)
        logits = rng.randn(6, 4).astype("float32")
        tgt = np.array([0, 1, 2, 3, 1, 0], dtype="int64")
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        ref = -logp[np.arange(6), tgt].sum()
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt), reduction="sum").numpy(),
                        ref, places=2, msg=f"cross_entropy sum {dev}")
        both_devices(body)

    def test_nll_loss(self):
        rng = np.random.RandomState(5)
        logits = rng.randn(5, 3).astype("float32")
        logp = (logits - np.log(np.exp(logits).sum(-1, keepdims=True))).astype("float32")
        tgt = np.array([2, 0, 1, 2, 0], dtype="int64")
        ref = -logp[np.arange(5), tgt].mean()
        def body(dev):
            self.scalar(F.nll_loss(t(logp), t(tgt)).numpy(), ref, msg=f"nll_loss {dev}")
        both_devices(body)

    def test_binary_cross_entropy(self):
        rng = np.random.RandomState(6)
        p = rng.rand(3, 4).astype("float32")
        y = (rng.rand(3, 4) > 0.5).astype("float32")
        ref = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        def body(dev):
            self.scalar(F.binary_cross_entropy(t(p), t(y)).numpy(), ref,
                        msg=f"bce {dev}")
        both_devices(body)

    def test_binary_cross_entropy_with_logits(self):
        rng = np.random.RandomState(7)
        logits = rng.randn(3, 4).astype("float32")
        y = (rng.rand(3, 4) > 0.5).astype("float32")
        # numerically-stable reference: max(l,0) - l*y + log(1+exp(-|l|))
        ref = (np.maximum(logits, 0) - logits * y
               + np.log1p(np.exp(-np.abs(logits)))).mean()
        def body(dev):
            self.scalar(F.binary_cross_entropy_with_logits(t(logits), t(y)).numpy(),
                        ref, msg=f"bce_logits {dev}")
        both_devices(body)

    def test_kl_div(self):
        rng = np.random.RandomState(8)
        logq = rng.randn(4, 5).astype("float32")
        inp = (logq - np.log(np.exp(logq).sum(-1, keepdims=True))).astype("float32")
        tgt = rng.rand(4, 5).astype("float32"); tgt /= tgt.sum(-1, keepdims=True)
        elem = tgt * (np.log(tgt) - inp)
        def body(dev):
            # torch default reduction='mean' divides the elementwise sum by NUMEL.
            self.scalar(F.kl_div(t(inp), t(tgt)).numpy(), elem.sum() / elem.size,
                        msg=f"kl_div mean {dev}")
            self.scalar(F.kl_div(t(inp), t(tgt), reduction="sum").numpy(), elem.sum(),
                        places=3, msg=f"kl_div sum {dev}")
            self.scalar(F.kl_div(t(inp), t(tgt), reduction="batchmean").numpy(),
                        elem.sum() / inp.shape[0], msg=f"kl_div batchmean {dev}")
        both_devices(body)


# ------------------------------------------------------------------------------ norms

class TestNorms(Base):
    def test_layer_norm(self):
        rng = np.random.RandomState(9)
        x = rng.randn(2, 3, 4).astype("float32")
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)             # biased (ddof=0), as torch layer_norm
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            self.ac(F.layer_norm(t(x), (4,)).numpy(), ref, atol=1e-4,
                    msg=f"layer_norm {dev}")
        both_devices(body)

    def test_layer_norm_affine(self):
        rng = np.random.RandomState(90)
        x = rng.randn(2, 5).astype("float32")
        w = rng.randn(5).astype("float32"); b = rng.randn(5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5) * w + b
        def body(dev):
            self.ac(F.layer_norm(t(x), (5,), t(w), t(b)).numpy(), ref, atol=1e-4,
                    msg=f"layer_norm affine {dev}")
        both_devices(body)

    def test_group_norm(self):
        rng = np.random.RandomState(10)
        x = rng.randn(2, 6, 4).astype("float32")   # N, C=6, spatial=4
        groups = 2                                  # -> 2 groups of 3 channels
        xr = x.reshape(2, groups, 3, 4)
        m = xr.mean((2, 3), keepdims=True); v = xr.var((2, 3), keepdims=True)
        ref = ((xr - m) / np.sqrt(v + 1e-5)).reshape(2, 6, 4)
        def body(dev):
            self.ac(F.group_norm(t(x), groups).numpy(), ref, atol=1e-4,
                    msg=f"group_norm {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------- modules

class TestModules(Base):
    def test_linear_module(self):
        rng = np.random.RandomState(11)
        x = rng.randn(4, 5).astype("float32")
        def body(dev):
            lin = nn.Linear(5, 3)
            out = lin(t(x))
            self.assertEqual(tuple(out.shape), (4, 3), f"Linear shape {dev}")
            # forward matches its own (weight, bias) explicitly
            w = lin.weight.numpy(); b = lin.bias.numpy()
            self.ac(out.numpy(), x @ w.T + b, atol=1e-4, msg=f"Linear fwd {dev}")
        both_devices(body)

    def test_relu_module(self):
        rng = np.random.RandomState(12)
        x = rng.randn(4, 5).astype("float32")
        def body(dev):
            self.ac(nn.ReLU()(t(x)).numpy(), np.maximum(x, 0), msg=f"ReLU mod {dev}")
        both_devices(body)

    def test_layernorm_module(self):
        rng = np.random.RandomState(13)
        x = rng.randn(4, 5).astype("float32")
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        # default affine: weight=1, bias=0 -> just the normalization
        ref = (x - mu) / np.sqrt(var + 1e-5)
        def body(dev):
            lnm = nn.LayerNorm(5)
            out = lnm(t(x))
            self.assertEqual(tuple(out.shape), (4, 5), f"LayerNorm shape {dev}")
            self.ac(out.numpy(), ref, atol=1e-4, msg=f"LayerNorm fwd {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
