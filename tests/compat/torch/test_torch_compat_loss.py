"""Torch-grade loss-function parity for ``import jittor as torch``.

Part of the torch-grade test-suite expansion. Like the sibling
``test_torch_compat_nn.py`` / ``test_torch_compat_math.py`` modules this is a structured
``unittest`` module: every check compares jittor-as-torch's ``nn.functional`` / ``nn``
loss criteria against an INDEPENDENT, explicit numpy / closed-form reference, and runs on
BOTH CPU and CUDA (when the build has it), so it locks torch *loss semantics* rather than
jittor self-consistency.

Covers losses not (or only lightly) exercised by test_torch_compat_nn.py:
  * ``cross_entropy``: reduction none/mean/sum, ``ignore_index`` (incl. the index-0 case),
    class ``weight``, ``label_smoothing``, the (N,C,H,W) spatial form, and SOFT
    (probability) targets (mixup / distillation) incl. weighted + smoothed.
  * ``nll_loss``: reduction, ``weight``, ``ignore_index`` (incl. index 0, the real bug).
  * ``binary_cross_entropy`` + ``..._with_logits``: ``weight`` / ``pos_weight`` and the
    full reduction set (none/sum/mean) on both the functional and the criterion class.
  * ``kl_div``: mean/sum/batchmean and ``log_target``.
  * ``smooth_l1_loss`` / ``huber_loss`` (incl. ``delta``/``beta`` != 1).
  * margin family: ``margin_ranking_loss`` / ``cosine_embedding_loss``.
  * criterion classes: CrossEntropyLoss(reduction=), NLLLoss, BCELoss, BCEWithLogitsLoss,
    KLDivLoss, SmoothL1Loss, HuberLoss, MarginRankingLoss, CosineEmbeddingLoss.

Notes:
  * jittor has no 0-d scalars; reduced losses are shape ``(1,)`` -> compared via the
    Python value against the numpy scalar.
  * torch's ``F.cross_entropy``/``F.nll_loss`` mean-reduction divides by the SUM of the
    weights of the non-ignored targets (= count of kept targets when unweighted); the
    references match that. For SOFT targets torch divides by N (batch numel) instead.

Run:  python -m pytest tests/compat/torch/test_torch_compat_loss.py
      python -m pytest tests/compat/torch/test_torch_compat_loss.py
"""
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


def np_logsoftmax(x, axis):
    return x - (np.log(np.exp(x - x.max(axis, keepdims=True)).sum(axis, keepdims=True))
                + x.max(axis, keepdims=True))


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def scalar(self, got, ref, places=4, msg=""):
        # jittor reduced loss is shape (1,); compare the python value.
        self.assertAlmostEqual(float(np.asarray(got).reshape(-1)[0]), float(ref),
                               places=places, msg=msg)


# ----------------------------------------------------------------------- cross_entropy

class TestCrossEntropy(Base):
    def setUp(self):
        rng = np.random.RandomState(100)
        self.logits = rng.randn(6, 4).astype("float32")
        self.tgt = np.array([0, 1, 2, 3, 2, 1], dtype="int64")
        self.logp = np_logsoftmax(self.logits, -1).astype("float32")

    def test_reduction_none(self):
        logits, tgt, logp = self.logits, self.tgt, self.logp
        ref = -logp[np.arange(6), tgt]
        def body(dev):
            self.ac(F.cross_entropy(t(logits), t(tgt), reduction="none").numpy(), ref,
                    atol=1e-5, msg=f"ce none {dev}")
        both_devices(body)

    def test_reduction_mean_sum(self):
        logits, tgt, logp = self.logits, self.tgt, self.logp
        per = -logp[np.arange(6), tgt]
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt)).numpy(), per.mean(),
                        msg=f"ce mean {dev}")
            self.scalar(F.cross_entropy(t(logits), t(tgt), reduction="sum").numpy(),
                        per.sum(), places=2, msg=f"ce sum {dev}")
        both_devices(body)

    def test_ignore_index(self):
        # torch ignores ANY class id >= 0; the index-0 case was the regression.
        logits, tgt, logp = self.logits, self.tgt, self.logp
        for ign in [0, 2, 3]:
            keep = tgt != ign
            ref = -logp[np.arange(6), tgt][keep].sum() / keep.sum()
            def body(dev, ign=ign, ref=ref):
                self.scalar(F.cross_entropy(t(logits), t(tgt), ignore_index=ign).numpy(),
                            ref, msg=f"ce ignore_index={ign} {dev}")
            both_devices(body)

    def test_weight(self):
        logits, tgt, logp = self.logits, self.tgt, self.logp
        w = np.array([1.0, 2.0, 0.5, 3.0], dtype="float32")
        num = -(logp[np.arange(6), tgt] * w[tgt]).sum()
        den = w[tgt].sum()
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt), weight=t(w)).numpy(),
                        num / den, msg=f"ce weight {dev}")
        both_devices(body)

    def test_label_smoothing(self):
        logits, tgt, logp = self.logits, self.tgt, self.logp
        ls, C = 0.1, 4
        onehot = np.eye(C, dtype="float32")[tgt]
        smooth_tgt = (1 - ls) * onehot + ls / C
        per = -(smooth_tgt * logp).sum(-1)
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt), label_smoothing=ls).numpy(),
                        per.mean(), msg=f"ce label_smoothing {dev}")
        both_devices(body)

    def test_spatial_4d(self):
        rng = np.random.RandomState(101)
        logits = rng.randn(2, 4, 3, 3).astype("float32")
        tgt = rng.randint(0, 4, size=(2, 3, 3)).astype("int64")
        lp = np_logsoftmax(logits, 1)
        per = -np.take_along_axis(lp, tgt[:, None], axis=1)[:, 0]
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(tgt)).numpy(), per.mean(),
                        msg=f"ce 4d mean {dev}")
            self.ac(F.cross_entropy(t(logits), t(tgt), reduction="none").numpy(), per,
                    atol=1e-5, msg=f"ce 4d none {dev}")
        both_devices(body)

    def test_soft_target(self):
        # torch: a float target with the same shape as input is a class-distribution
        # target (mixup / distillation). mean-reduction divides by N here, not by Σw.
        rng = np.random.RandomState(102)
        logits = rng.randn(5, 3).astype("float32")
        soft = rng.rand(5, 3).astype("float32"); soft /= soft.sum(-1, keepdims=True)
        lp = np_logsoftmax(logits, -1)
        per = -(soft * lp).sum(-1)
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(soft)).numpy(), per.mean(),
                        msg=f"ce soft mean {dev}")
            self.scalar(F.cross_entropy(t(logits), t(soft), reduction="sum").numpy(),
                        per.sum(), places=3, msg=f"ce soft sum {dev}")
            self.ac(F.cross_entropy(t(logits), t(soft), reduction="none").numpy(), per,
                    atol=1e-5, msg=f"ce soft none {dev}")
        both_devices(body)

    def test_soft_target_weight_smoothing(self):
        rng = np.random.RandomState(103)
        logits = rng.randn(5, 3).astype("float32")
        soft = rng.rand(5, 3).astype("float32"); soft /= soft.sum(-1, keepdims=True)
        w = np.array([1.0, 2.0, 0.5], dtype="float32")
        lp = np_logsoftmax(logits, -1)
        per_w = -(soft * lp * w.reshape(1, 3)).sum(-1)
        ls = 0.1
        tgt_s = (1 - ls) * soft + ls / 3
        per_ls = -(tgt_s * lp).sum(-1)
        def body(dev):
            self.scalar(F.cross_entropy(t(logits), t(soft), weight=t(w)).numpy(),
                        per_w.mean(), msg=f"ce soft weight {dev}")
            self.scalar(F.cross_entropy(t(logits), t(soft), label_smoothing=ls).numpy(),
                        per_ls.mean(), msg=f"ce soft smoothing {dev}")
        both_devices(body)

    def test_criterion_class_reduction(self):
        # torch.nn.CrossEntropyLoss takes reduction=; it used to be silently dropped.
        logits, tgt, logp = self.logits, self.tgt, self.logp
        per = -logp[np.arange(6), tgt]
        def body(dev):
            self.scalar(nn.CrossEntropyLoss(reduction="sum")(t(logits), t(tgt)).numpy(),
                        per.sum(), places=2, msg=f"CELoss sum {dev}")
            self.ac(nn.CrossEntropyLoss(reduction="none")(t(logits), t(tgt)).numpy(),
                    per, atol=1e-5, msg=f"CELoss none {dev}")
            self.scalar(nn.CrossEntropyLoss()(t(logits), t(tgt)).numpy(), per.mean(),
                        msg=f"CELoss mean {dev}")
        both_devices(body)


# --------------------------------------------------------------------------- nll_loss

class TestNLLLoss(Base):
    def setUp(self):
        rng = np.random.RandomState(110)
        logits = rng.randn(6, 4).astype("float32")
        self.logp = np_logsoftmax(logits, -1).astype("float32")
        self.tgt = np.array([0, 1, 2, 3, 2, 1], dtype="int64")

    def test_basic_reductions(self):
        logp, tgt = self.logp, self.tgt
        per = -logp[np.arange(6), tgt]
        def body(dev):
            self.scalar(F.nll_loss(t(logp), t(tgt)).numpy(), per.mean(),
                        msg=f"nll mean {dev}")
            self.scalar(F.nll_loss(t(logp), t(tgt), reduction="sum").numpy(), per.sum(),
                        places=3, msg=f"nll sum {dev}")
            self.ac(F.nll_loss(t(logp), t(tgt), reduction="none").numpy(), per,
                    atol=1e-5, msg=f"nll none {dev}")
        both_devices(body)

    def test_ignore_index_including_zero(self):
        # The real bug: ignore_index=0 was silently NOT honored (old `>0` guard).
        logp, tgt = self.logp, self.tgt
        for ign in [0, 1, 3]:
            keep = tgt != ign
            ref = -logp[np.arange(6), tgt][keep].sum() / keep.sum()
            def body(dev, ign=ign, ref=ref):
                self.scalar(F.nll_loss(t(logp), t(tgt), ignore_index=ign).numpy(), ref,
                            msg=f"nll ignore_index={ign} {dev}")
            both_devices(body)

    def test_weight_and_ignore(self):
        logp, tgt = self.logp, self.tgt
        w = np.array([1.0, 2.0, 0.5, 3.0], dtype="float32")
        for ign in [-100, 0, 2]:
            keep = tgt != ign
            num = -(logp[np.arange(6), tgt] * w[tgt])[keep].sum()
            den = (w[tgt] * keep).sum()
            def body(dev, ign=ign, num=num, den=den):
                self.scalar(F.nll_loss(t(logp), t(tgt), weight=t(w),
                                       ignore_index=ign).numpy(), num / den,
                            msg=f"nll weight ignore={ign} {dev}")
            both_devices(body)

    def test_weight_not_mutated(self):
        # ignore_index zeroing must not corrupt a reused weight Var.
        logp, tgt = self.logp, self.tgt
        def body(dev):
            w = torch.ones((4,))
            F.nll_loss(t(logp), t(tgt), weight=w, ignore_index=0)
            self.ac(w.numpy(), np.ones(4, dtype="float32"), msg=f"nll weight intact {dev}")
        both_devices(body)


# ----------------------------------------------------------------- binary cross entropy

class TestBCE(Base):
    def setUp(self):
        rng = np.random.RandomState(120)
        self.p = rng.rand(3, 4).astype("float32") * 0.98 + 0.01
        self.y = (rng.rand(3, 4) > 0.5).astype("float32")
        self.logits = rng.randn(3, 4).astype("float32")

    def test_bce_reductions(self):
        p, y = self.p, self.y
        per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        def body(dev):
            self.scalar(F.binary_cross_entropy(t(p), t(y)).numpy(), per.mean(),
                        msg=f"bce mean {dev}")
            self.scalar(F.binary_cross_entropy(t(p), t(y), reduction="sum").numpy(),
                        per.sum(), places=3, msg=f"bce sum {dev}")
            self.ac(F.binary_cross_entropy(t(p), t(y), reduction="none").numpy(), per,
                    atol=1e-5, msg=f"bce none {dev}")
        both_devices(body)

    def test_bce_weight(self):
        rng = np.random.RandomState(121)
        p, y = self.p, self.y
        w = rng.rand(3, 4).astype("float32")
        per = -(y * np.log(p) + (1 - y) * np.log(1 - p)) * w
        def body(dev):
            self.scalar(F.binary_cross_entropy(t(p), t(y), weight=t(w)).numpy(),
                        per.mean(), msg=f"bce weight {dev}")
        both_devices(body)

    def test_bce_logits_reductions(self):
        logits, y = self.logits, self.y
        sig = 1.0 / (1.0 + np.exp(-logits))
        per = -(y * np.log(sig) + (1 - y) * np.log(1 - sig))
        def body(dev):
            self.scalar(F.binary_cross_entropy_with_logits(t(logits), t(y)).numpy(),
                        per.mean(), msg=f"bce_logits mean {dev}")
            self.scalar(F.binary_cross_entropy_with_logits(
                t(logits), t(y), reduction="sum").numpy(), per.sum(), places=3,
                msg=f"bce_logits sum {dev}")
            self.ac(F.binary_cross_entropy_with_logits(
                t(logits), t(y), reduction="none").numpy(), per, atol=1e-5,
                msg=f"bce_logits none {dev}")
        both_devices(body)

    def test_bce_logits_pos_weight(self):
        logits, y = self.logits, self.y
        pw = np.array([1.0, 2.0, 0.5, 3.0], dtype="float32")
        sig = 1.0 / (1.0 + np.exp(-logits))
        per = -(pw * y * np.log(sig) + (1 - y) * np.log(1 - sig))
        def body(dev):
            self.scalar(F.binary_cross_entropy_with_logits(
                t(logits), t(y), pos_weight=t(pw)).numpy(), per.mean(),
                msg=f"bce_logits pos_weight {dev}")
        both_devices(body)

    def test_bce_logits_class(self):
        logits, y = self.logits, self.y
        sig = 1.0 / (1.0 + np.exp(-logits))
        per = -(y * np.log(sig) + (1 - y) * np.log(1 - sig))
        def body(dev):
            self.scalar(nn.BCEWithLogitsLoss(reduction="sum")(t(logits), t(y)).numpy(),
                        per.sum(), places=3, msg=f"BCEWithLogitsLoss sum {dev}")
            self.scalar(nn.BCEWithLogitsLoss()(t(logits), t(y)).numpy(), per.mean(),
                        msg=f"BCEWithLogitsLoss mean {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------- kl_div

class TestKLDiv(Base):
    def setUp(self):
        rng = np.random.RandomState(130)
        logq = rng.randn(4, 5).astype("float32")
        self.inp = np_logsoftmax(logq, -1).astype("float32")
        tgt = rng.rand(4, 5).astype("float32"); tgt /= tgt.sum(-1, keepdims=True)
        self.tgt = tgt

    def test_reductions(self):
        inp, tgt = self.inp, self.tgt
        elem = tgt * (np.log(tgt) - inp)
        def body(dev):
            self.scalar(F.kl_div(t(inp), t(tgt)).numpy(), elem.sum() / elem.size,
                        msg=f"kl mean {dev}")
            self.scalar(F.kl_div(t(inp), t(tgt), reduction="sum").numpy(), elem.sum(),
                        places=3, msg=f"kl sum {dev}")
            self.scalar(F.kl_div(t(inp), t(tgt), reduction="batchmean").numpy(),
                        elem.sum() / inp.shape[0], msg=f"kl batchmean {dev}")
        both_devices(body)

    def test_log_target(self):
        rng = np.random.RandomState(131)
        inp = self.inp
        logt = np_logsoftmax(rng.randn(4, 5).astype("float32"), -1).astype("float32")
        elem = np.exp(logt) * (logt - inp)
        def body(dev):
            self.scalar(F.kl_div(t(inp), t(logt), log_target=True).numpy(),
                        elem.sum() / elem.size, msg=f"kl log_target {dev}")
        both_devices(body)

    def test_criterion_class(self):
        inp, tgt = self.inp, self.tgt
        elem = tgt * (np.log(tgt) - inp)
        def body(dev):
            self.scalar(nn.KLDivLoss(reduction="batchmean")(t(inp), t(tgt)).numpy(),
                        elem.sum() / inp.shape[0], msg=f"KLDivLoss batchmean {dev}")
            self.scalar(nn.KLDivLoss(reduction="sum")(t(inp), t(tgt)).numpy(),
                        elem.sum(), places=3, msg=f"KLDivLoss sum {dev}")
        both_devices(body)


# ------------------------------------------------------------------ smooth_l1 / huber

class TestSmoothL1Huber(Base):
    def setUp(self):
        rng = np.random.RandomState(140)
        self.a = (rng.randn(4, 5) * 1.5).astype("float32")
        self.b = (rng.randn(4, 5) * 1.5).astype("float32")

    def test_smooth_l1(self):
        a, b = self.a, self.b
        d = np.abs(a - b)
        per = np.where(d < 1.0, 0.5 * d * d, d - 0.5)   # torch beta=1
        def body(dev):
            self.ac(F.smooth_l1_loss(t(a), t(b), reduction="none").numpy(), per,
                    atol=1e-5, msg=f"smooth_l1 none {dev}")
            self.scalar(F.smooth_l1_loss(t(a), t(b)).numpy(), per.mean(),
                        msg=f"smooth_l1 mean {dev}")
            self.scalar(F.smooth_l1_loss(t(a), t(b), reduction="sum").numpy(), per.sum(),
                        places=3, msg=f"smooth_l1 sum {dev}")
        both_devices(body)

    def test_huber_delta(self):
        a, b = self.a, self.b
        d = np.abs(a - b)
        for delta in [1.0, 2.0, 0.5]:
            per = np.where(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta))
            def body(dev, delta=delta, per=per):
                self.ac(F.huber_loss(t(a), t(b), reduction="none", delta=delta).numpy(),
                        per, atol=1e-5, msg=f"huber delta={delta} none {dev}")
                self.scalar(F.huber_loss(t(a), t(b), delta=delta).numpy(), per.mean(),
                            msg=f"huber delta={delta} mean {dev}")
            both_devices(body)

    def test_huber_class(self):
        a, b = self.a, self.b
        d = np.abs(a - b)
        per = np.where(d < 2.0, 0.5 * d * d, 2.0 * (d - 1.0))
        def body(dev):
            self.scalar(nn.HuberLoss(delta=2.0)(t(a), t(b)).numpy(), per.mean(),
                        msg=f"HuberLoss delta=2 {dev}")
        both_devices(body)


# ------------------------------------------------------------------------ margin family

class TestMargin(Base):
    def test_margin_ranking(self):
        rng = np.random.RandomState(150)
        x1 = rng.randn(6).astype("float32"); x2 = rng.randn(6).astype("float32")
        y = np.array([1, -1, 1, -1, 1, -1], dtype="float32")
        for margin in [0.0, 0.3]:
            per = np.maximum(-y * (x1 - x2) + margin, 0.0)
            def body(dev, margin=margin, per=per):
                self.scalar(F.margin_ranking_loss(t(x1), t(x2), t(y),
                                                  margin=margin).numpy(), per.mean(),
                            msg=f"margin_ranking margin={margin} {dev}")
                self.ac(F.margin_ranking_loss(t(x1), t(x2), t(y), margin=margin,
                                              reduction="none").numpy(), per, atol=1e-5,
                        msg=f"margin_ranking none {dev}")
            both_devices(body)

    def test_cosine_embedding(self):
        rng = np.random.RandomState(151)
        a = rng.randn(4, 6).astype("float32"); b = rng.randn(4, 6).astype("float32")
        y = np.array([1, -1, 1, -1], dtype="float32")
        margin = 0.2
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        per = np.where(y == 1, 1.0 - cos, np.maximum(cos - margin, 0.0))
        def body(dev):
            self.scalar(F.cosine_embedding_loss(t(a), t(b), t(y), margin=margin).numpy(),
                        per.mean(), places=4, msg=f"cosine_embedding {dev}")
        both_devices(body)

    def test_margin_ranking_class(self):
        rng = np.random.RandomState(152)
        x1 = rng.randn(5).astype("float32"); x2 = rng.randn(5).astype("float32")
        y = np.array([1, -1, 1, 1, -1], dtype="float32")
        per = np.maximum(-y * (x1 - x2) + 0.5, 0.0)
        def body(dev):
            self.scalar(nn.MarginRankingLoss(margin=0.5)(t(x1), t(x2), t(y)).numpy(),
                        per.mean(), msg=f"MarginRankingLoss {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
