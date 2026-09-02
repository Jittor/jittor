"""Torch-parity regression tests for the average-pooling family (bug #11).

Locks two PyTorch-documented behaviours that ``import jittor as torch`` must match,
each checked against an INDEPENDENT numpy reference computed from torch's *documented*
formula (no scipy / no real torch assumed), on BOTH CPU and CUDA:

  1. ``F.avg_pool2d(..., padding>0, count_include_pad=True|False)``
       torch DEFAULT count_include_pad=True  -> padded zeros ARE in the denominator
                                                (divisor = full kernel area, clamped
                                                 to the padded input under ceil_mode).
       count_include_pad=False               -> only real (in-bounds) input elements
                                                 count toward the denominator.
     (jittor.pool.AvgPool2d used to ignore the flag; it forwards to
      nn.AvgPool2d now -- see tests/nn/test_avg_pool_parity.py.)

  2. ``F.adaptive_avg_pool2d(x, out)`` with out NOT dividing the input size.
       torch uses variable-width overlapping bins
         hstart=floor(i*H/Oh), hend=ceil((i+1)*H/Oh)  (and likewise for W)
       dividing by the real bin area. jittor.pool used a single uniform stride/kernel,
       which agrees only when out | input; nn.AdaptiveAvgPool2d is corrected.

Run:  python -m pytest tests/compat/torch/test_torch_compat_pool_parity.py
      python -m pytest tests/compat/torch/test_torch_compat_pool_parity.py
"""
import math
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt
import jittor.nn as nn

F = nn.functional

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


# ---------------------------------------------------------------------------
# Independent numpy references, derived straight from torch's documented rules.
# ---------------------------------------------------------------------------
def avg_pool2d_ref(x, k, stride=None, padding=0, ceil_mode=False,
                   count_include_pad=True):
    """Reference avg_pool2d matching torch's documented divisor semantics.

    The padded array spans rows ``[0, H+2*ph)``; ``count_include_pad=True`` divides by
    the window area clamped to that padded array (so ceil_mode overshoot past the right
    pad is NOT counted), while ``False`` divides by the area clamped to the real input
    ``[ph, ph+H)``.
    """
    kh, kw = _pair(k)
    sh, sw = _pair(k if stride is None else stride)
    ph, pw = _pair(padding)
    x = x.astype(np.float64)
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=0.0)
    Hp, Wp = xp.shape[2], xp.shape[3]
    rnd = math.ceil if ceil_mode else math.floor
    Ho = int(rnd((Hp - kh) / sh)) + 1
    Wo = int(rnd((Wp - kw) / sw)) + 1
    if ceil_mode:
        if (Ho - 1) * sh >= Hp + ph:
            Ho -= 1
        if (Wo - 1) * sw >= Wp + pw:
            Wo -= 1
    out = np.zeros((N, C, Ho, Wo))
    for n in range(N):
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    hs, ws = i * sh, j * sw
                    he, we = min(hs + kh, Hp), min(ws + kw, Wp)
                    win = xp[n, c, hs:he, ws:we]
                    if count_include_pad:
                        denom = (he - hs) * (we - ws)
                    else:
                        hs_r, ws_r = max(hs, ph), max(ws, pw)
                        he_r, we_r = min(hs + kh, ph + H), min(ws + kw, pw + W)
                        denom = (he_r - hs_r) * (we_r - ws_r)
                    out[n, c, i, j] = win.sum() / denom
    return out


def adaptive_avg_ref(x, out_size):
    """torch adaptive-avg: variable bins floor(i*H/O) .. ceil((i+1)*H/O), mean of bin."""
    Oh, Ow = _pair(out_size)
    x = x.astype(np.float64)
    N, C, H, W = x.shape
    out = np.zeros((N, C, Oh, Ow))
    for i in range(Oh):
        hs = (i * H) // Oh
        he = ((i + 1) * H + Oh - 1) // Oh
        for j in range(Ow):
            ws = (j * W) // Ow
            we = ((j + 1) * W + Ow - 1) // Ow
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(axis=(2, 3))
    return out


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-4, rtol=1e-4, msg=""):
        g = np.asarray(got)
        r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape),
                         f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# ---------------------------------------------------------------------------
# Gap 1: F.avg_pool2d count_include_pad (True default, and False).
# ---------------------------------------------------------------------------
class TestAvgPoolCountIncludePad(Base):
    def setUp(self):
        self.x = np.random.RandomState(2).randn(2, 3, 8, 8).astype("float32")

    def test_count_include_pad_true_is_default(self):
        # torch default: padded zeros counted -> divisor = full kernel area.
        x = self.x

        def body(dev):
            d = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1).numpy()
            self.ac(d, avg_pool2d_ref(x, 3, 2, 1, count_include_pad=True),
                    msg=f"avgpool default(incl) {dev}")
            e = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                             count_include_pad=True).numpy()
            self.ac(e, d, msg=f"avgpool explicit incl==default {dev}")
        both_devices(body)

    def test_count_include_pad_false_uses_real_element_count(self):
        # The headline gap: divisor = count of real (in-bounds) elements only.
        x = self.x

        def body(dev):
            g = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                             count_include_pad=False).numpy()
            self.ac(g, avg_pool2d_ref(x, 3, 2, 1, count_include_pad=False),
                    msg=f"avgpool excl {dev}")
        both_devices(body)

    def test_incl_and_excl_actually_differ_at_padded_border(self):
        # Guard against a regression that silently makes the flag a no-op again:
        # with padding>0 the two must NOT be identical (corners differ).
        x = self.x

        def body(dev):
            incl = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                                count_include_pad=True).numpy()
            excl = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                                count_include_pad=False).numpy()
            self.assertFalse(np.allclose(incl, excl),
                             f"count_include_pad False must change padded borders {dev}")
        both_devices(body)

    def test_count_include_pad_various_configs(self):
        x = self.x
        cfgs = [(3, 2, 1), (2, 2, 1), (3, 1, 1), (4, 2, 2), (2, 1, 1)]
        for k, s, p in cfgs:
            for cip in (True, False):
                def body(dev, k=k, s=s, p=p, cip=cip):
                    g = F.avg_pool2d(torch.tensor(x), k, stride=s, padding=p,
                                     count_include_pad=cip).numpy()
                    self.ac(g, avg_pool2d_ref(x, k, s, p, count_include_pad=cip),
                            msg=f"avgpool k{k}s{s}p{p}cip{cip} {dev}")
                both_devices(body)

    def test_count_include_pad_with_ceil_mode(self):
        # ceil_mode overshoot past the input must never be treated as padding.
        x = self.x
        for cip in (True, False):
            def body(dev, cip=cip):
                g = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                                 ceil_mode=True, count_include_pad=cip).numpy()
                self.ac(g, avg_pool2d_ref(x, 3, 2, 1, ceil_mode=True,
                                          count_include_pad=cip),
                        msg=f"avgpool ceil cip{cip} {dev}")
            both_devices(body)

    def test_no_padding_default_still_divides_by_kernel(self):
        # Sanity: with padding=0 the divisor is the full kernel area both ways.
        x = self.x
        for k, s in [(2, None), (2, 2), (4, 4)]:
            def body(dev, k=k, s=s):
                g = F.avg_pool2d(torch.tensor(x), k, stride=s).numpy()
                self.ac(g, avg_pool2d_ref(x, k, s, 0), msg=f"avgpool k{k}s{s} {dev}")
            both_devices(body)

    def test_module_avgpool2d_matches_functional(self):
        x = self.x

        def body(dev):
            m = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
            g = m(torch.tensor(x)).numpy()
            self.ac(g, avg_pool2d_ref(x, 3, 2, 1, count_include_pad=False),
                    msg=f"AvgPool2d module excl {dev}")
        both_devices(body)

    def test_count_include_pad_backward_matches_numpy(self):
        # Backward must stay correct too (reindex+reduce is differentiable):
        # d(sum(out))/dx is, per input cell, the sum over windows covering it of
        # 1/denom(window). Check excl (the harder, non-uniform-denominator) case.
        x = self.x
        k, s, p = 3, 2, 1

        def ref_grad(count_include_pad):
            N, C, H, W = x.shape
            ref = avg_pool2d_ref(np.zeros_like(x), k, s, p,
                                 count_include_pad=count_include_pad)  # shape only
            Ho, Wo = ref.shape[2], ref.shape[3]
            g = np.zeros((N, C, H, W))
            for i in range(Ho):
                for j in range(Wo):
                    hs, ws = i * s - p, j * s - p
                    he, we = hs + k, ws + k
                    hs_c, ws_c = max(hs, 0), max(ws, 0)
                    he_c, we_c = min(he, H), min(we, W)
                    if count_include_pad:
                        denom = (min(he, H + p) - max(hs, -p)) * \
                                (min(we, W + p) - max(ws, -p))
                    else:
                        denom = (he_c - hs_c) * (we_c - ws_c)
                    g[:, :, hs_c:he_c, ws_c:we_c] += 1.0 / denom
            return g

        def body(dev):
            for cip in (True, False):
                xj = torch.tensor(x)
                y = F.avg_pool2d(xj, k, stride=s, padding=p,
                                 count_include_pad=cip).sum()
                ga = jt.grad(y, [xj])[0].numpy()
                self.ac(ga, ref_grad(cip), atol=1e-3, rtol=1e-3,
                        msg=f"avgpool grad cip{cip} {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# Gap 2: F.adaptive_avg_pool2d non-divisor output sizes.
# ---------------------------------------------------------------------------
class TestAdaptiveAvgPoolNonDivisor(Base):
    def test_nondivisor_square(self):
        # The classic diffusers / classifier-head case (8->3, plus more).
        for H, O in [(8, 3), (8, 5), (8, 6), (10, 3), (10, 4), (9, 4), (13, 5), (7, 4)]:
            x = np.random.RandomState(3).randn(2, 3, H, H).astype("float32")
            def body(dev, x=x, O=O):
                g = F.adaptive_avg_pool2d(torch.tensor(x), O).numpy()
                self.ac(g, adaptive_avg_ref(x, O), msg=f"adaptive {x.shape}->{O} {dev}")
            both_devices(body)

    def test_nondivisor_rectangular(self):
        for (H, W), O in [((10, 12), (5, 6)), ((7, 9), (3, 4)),
                          ((8, 8), (3, 5)), ((11, 13), (4, 4)), ((5, 7), (2, 3))]:
            x = np.random.RandomState(4).randn(2, 3, H, W).astype("float32")
            def body(dev, x=x, O=O):
                g = F.adaptive_avg_pool2d(torch.tensor(x), O).numpy()
                self.ac(g, adaptive_avg_ref(x, O),
                        msg=f"adaptive {x.shape}->{O} {dev}")
            both_devices(body)

    def test_divisor_sizes_still_match(self):
        # Must not regress the easy (output | input) cases the old impl already did.
        x = np.random.RandomState(3).randn(2, 3, 8, 8).astype("float32")
        for O in [1, 2, 4, 8]:
            def body(dev, O=O):
                g = F.adaptive_avg_pool2d(torch.tensor(x), O).numpy()
                self.ac(g, adaptive_avg_ref(x, O), msg=f"adaptive divisor {O} {dev}")
            both_devices(body)

    def test_output_one_is_global_mean(self):
        x = np.random.RandomState(5).randn(2, 3, 7, 11).astype("float32")
        def body(dev):
            g = F.adaptive_avg_pool2d(torch.tensor(x), 1).numpy()
            self.ac(g, x.astype(np.float64).mean(axis=(2, 3), keepdims=True),
                    msg=f"adaptive global mean {dev}")
        both_devices(body)

    def test_module_form_matches(self):
        x = np.random.RandomState(6).randn(2, 3, 10, 10).astype("float32")
        def body(dev):
            g = nn.AdaptiveAvgPool2d((3, 3))(torch.tensor(x)).numpy()
            self.ac(g, adaptive_avg_ref(x, 3), msg=f"AdaptiveAvgPool2d module {dev}")
        both_devices(body)

    def test_none_in_output_keeps_dim(self):
        # torch: output_size=(None, W) keeps H unchanged, pools only W.
        x = np.random.RandomState(7).randn(2, 3, 6, 8).astype("float32")
        def body(dev):
            g = F.adaptive_avg_pool2d(torch.tensor(x), (None, 3)).numpy()
            self.ac(g, adaptive_avg_ref(x, (6, 3)), msg=f"adaptive (None,3) {dev}")
        both_devices(body)

    def test_nondivisor_backward_matches_numpy(self):
        # Variable-bin backward: each input cell receives 1/binarea from every output
        # bin that covers it.
        x = np.random.RandomState(8).randn(2, 3, 8, 8).astype("float32")
        O = 3
        N, C, H, W = x.shape

        def ref_grad():
            g = np.zeros((N, C, H, W))
            for i in range(O):
                hs = (i * H) // O
                he = ((i + 1) * H + O - 1) // O
                for j in range(O):
                    ws = (j * W) // O
                    we = ((j + 1) * W + O - 1) // O
                    g[:, :, hs:he, ws:we] += 1.0 / ((he - hs) * (we - ws))
            return g

        def body(dev):
            xj = torch.tensor(x)
            y = F.adaptive_avg_pool2d(xj, O).sum()
            ga = jt.grad(y, [xj])[0].numpy()
            self.ac(ga, ref_grad(), atol=1e-3, rtol=1e-3,
                    msg=f"adaptive grad {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
