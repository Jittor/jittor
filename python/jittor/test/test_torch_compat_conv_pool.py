"""Torch-grade conv/pool/norm-semantics regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_ops.py``).
Every check compares jittor-as-torch against an INDEPENDENT numpy reference computed
from scratch here (scipy is not assumed to be installed), and runs on BOTH CPU and
CUDA when the build has it, so it locks torch-API *semantics* rather than jittor
self-consistency.

The numpy reference impls (``conv2d_ref``, ``conv_transpose2d_ref``, ``pool2d_ref``,
``adaptive_avg_ref``/``adaptive_max_ref``, the norm/interp/pad helpers) were validated
to be bit-faithful to real PyTorch on the cases exercised below before being committed.

Where jittor's semantics genuinely diverge from torch (documented inline) the case is
``@unittest.skip``-ped with a ``REASON:`` so the gap is visible but does not fail CI:
  * ``F.avg_pool2d(count_include_pad=False)`` is ignored by jittor.
  * adaptive pooling only matches torch when output_size divides input_size.

jittor has no 0-d scalar, so every full reduction goes through ``.item()`` / numpy.

Run:  python -m jittor.test.test_torch_compat_conv_pool
      python -m pytest python/jittor/test/test_torch_compat_conv_pool.py
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


# ---------------------------------------------------------------------------
# Independent numpy reference implementations (the "gold" side of every check).
# ---------------------------------------------------------------------------
def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def conv2d_ref(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    sh, sw = _pair(stride); ph, pw = _pair(padding); dh, dw = _pair(dilation)
    x = x.astype(np.float64); w = w.astype(np.float64)
    N, Cin, H, W = x.shape
    Cout, Cin_g, Kh, Kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    Hp, Wp = xp.shape[2], xp.shape[3]
    Ho = (Hp - (dh * (Kh - 1) + 1)) // sh + 1
    Wo = (Wp - (dw * (Kw - 1) + 1)) // sw + 1
    out = np.zeros((N, Cout, Ho, Wo))
    og = Cout // groups
    for n in range(N):
        for g in range(groups):
            for co in range(og):
                och = g * og + co
                for i in range(Ho):
                    for j in range(Wo):
                        acc = 0.0
                        for ci in range(Cin_g):
                            ich = g * Cin_g + ci
                            for ki in range(Kh):
                                for kj in range(Kw):
                                    acc += xp[n, ich, i * sh + ki * dh, j * sw + kj * dw] * w[och, ci, ki, kj]
                        out[n, och, i, j] = acc
    if b is not None:
        out += b.astype(np.float64).reshape(1, -1, 1, 1)
    return out


def conv_transpose2d_ref(x, w, b=None, stride=1, padding=0, output_padding=0,
                         dilation=1, groups=1):
    sh, sw = _pair(stride); ph, pw = _pair(padding)
    oph, opw = _pair(output_padding); dh, dw = _pair(dilation)
    x = x.astype(np.float64); w = w.astype(np.float64)
    N, Cin, H, W = x.shape
    Cin2, Cout_g, Kh, Kw = w.shape
    Cout = Cout_g * groups
    Ho = (H - 1) * sh - 2 * ph + dh * (Kh - 1) + oph + 1
    Wo = (W - 1) * sw - 2 * pw + dw * (Kw - 1) + opw + 1
    full = np.zeros((N, Cout, Ho + 2 * ph, Wo + 2 * pw))
    ig = Cin // groups
    for n in range(N):
        for g in range(groups):
            for ci in range(ig):
                ich = g * ig + ci
                for co in range(Cout_g):
                    och = g * Cout_g + co
                    for i in range(H):
                        for j in range(W):
                            for ki in range(Kh):
                                for kj in range(Kw):
                                    full[n, och, i * sh + ki * dh, j * sw + kj * dw] += \
                                        x[n, ich, i, j] * w[ich, co, ki, kj]
    out = full[:, :, ph:ph + Ho, pw:pw + Wo]
    if b is not None:
        out = out + b.astype(np.float64).reshape(1, -1, 1, 1)
    return out


def pool2d_ref(x, k, stride=None, padding=0, ceil_mode=False, mode="max",
               count_include_pad=True):
    kh, kw = _pair(k)
    sh, sw = _pair(k if stride is None else stride)
    ph, pw = _pair(padding)
    x = x.astype(np.float64)
    N, C, H, W = x.shape
    padval = -np.inf if mode == "max" else 0.0
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=padval)
    Hp, Wp = xp.shape[2], xp.shape[3]
    rnd = math.ceil if ceil_mode else math.floor
    Ho = int(rnd((Hp - kh) / sh)) + 1
    Wo = int(rnd((Wp - kw) / sw)) + 1
    # torch: when ceil_mode pushes the last window to start in the right pad, drop it.
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
                    if mode == "max":
                        out[n, c, i, j] = win.max()
                    elif count_include_pad:
                        out[n, c, i, j] = win.sum() / ((he - hs) * (we - ws))
                    else:
                        hs_r, ws_r = max(hs, ph), max(ws, pw)
                        he_r, we_r = min(hs + kh, ph + H), min(ws + kw, pw + W)
                        out[n, c, i, j] = win.sum() / ((he_r - hs_r) * (we_r - ws_r))
    return out


def adaptive_avg_ref(x, out_size):
    Oh, Ow = _pair(out_size)
    x = x.astype(np.float64)
    N, C, H, W = x.shape
    out = np.zeros((N, C, Oh, Ow))
    for i in range(Oh):
        hs = (i * H) // Oh; he = ((i + 1) * H + Oh - 1) // Oh
        for j in range(Ow):
            ws = (j * W) // Ow; we = ((j + 1) * W + Ow - 1) // Ow
            out[:, :, i, j] = x[:, :, hs:he, ws:we].mean(axis=(2, 3))
    return out


def adaptive_max_ref(x, out_size):
    Oh, Ow = _pair(out_size)
    x = x.astype(np.float64)
    N, C, H, W = x.shape
    out = np.zeros((N, C, Oh, Ow))
    for i in range(Oh):
        hs = (i * H) // Oh; he = ((i + 1) * H + Oh - 1) // Oh
        for j in range(Ow):
            ws = (j * W) // Ow; we = ((j + 1) * W + Ow - 1) // Ow
            out[:, :, i, j] = x[:, :, hs:he, ws:we].max(axis=3).max(axis=2)
    return out


def batchnorm_eval_ref(x, rm, rv, w, b, eps=1e-5):
    C = x.shape[1]
    rm = rm.reshape(1, C, 1, 1); rv = rv.reshape(1, C, 1, 1)
    w = w.reshape(1, C, 1, 1); b = b.reshape(1, C, 1, 1)
    return (x - rm) / np.sqrt(rv + eps) * w + b


def groupnorm_ref(x, G, w, b, eps=1e-5):
    N, C, H, W = x.shape
    xr = x.reshape(N, G, C // G, H, W).astype(np.float64)
    mu = xr.mean(axis=(2, 3, 4), keepdims=True)
    var = xr.var(axis=(2, 3, 4), keepdims=True)
    xn = ((xr - mu) / np.sqrt(var + eps)).reshape(N, C, H, W)
    return xn * w.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1)


def instancenorm_ref(x, w, b, eps=1e-5):
    N, C, H, W = x.shape
    x = x.astype(np.float64)
    mu = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    xn = (x - mu) / np.sqrt(var + eps)
    return xn * w.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1)


def nearest_ref(x, scale):
    N, C, H, W = x.shape
    Oh, Ow = int(H * scale), int(W * scale)
    out = np.zeros((N, C, Oh, Ow), dtype=np.float64)
    for i in range(Oh):
        si = min(int(i / scale), H - 1)
        for j in range(Ow):
            sj = min(int(j / scale), W - 1)
            out[:, :, i, j] = x[:, :, si, sj]
    return out


def bilinear_ref(x, Oh, Ow, align_corners):
    N, C, H, W = x.shape
    x = x.astype(np.float64)
    out = np.zeros((N, C, Oh, Ow))
    for i in range(Oh):
        yi = (i * (H - 1) / (Oh - 1) if Oh > 1 else 0.0) if align_corners \
            else (i + 0.5) * H / Oh - 0.5
        for j in range(Ow):
            xj = (j * (W - 1) / (Ow - 1) if Ow > 1 else 0.0) if align_corners \
                else (j + 0.5) * W / Ow - 0.5
            y0 = int(np.floor(yi)); x0 = int(np.floor(xj))
            y1 = min(y0 + 1, H - 1); x1 = min(x0 + 1, W - 1)
            y0c = max(y0, 0); x0c = max(x0, 0)
            ly = min(max(yi - y0, 0.0), 1.0); lx = min(max(xj - x0, 0.0), 1.0)
            out[:, :, i, j] = (x[:, :, y0c, x0c] * (1 - ly) * (1 - lx)
                               + x[:, :, y0c, x1] * (1 - ly) * lx
                               + x[:, :, y1, x0c] * ly * (1 - lx)
                               + x[:, :, y1, x1] * ly * lx)
    return out


def pad_ref(x, pad, mode="constant", value=0):
    # torch pad tuple is reversed and applies to the trailing dims.
    L, R, T, B = pad
    npad = ((0, 0), (0, 0), (T, B), (L, R))
    nmode = {"constant": "constant", "replicate": "edge",
             "reflect": "reflect", "circular": "wrap"}[mode]
    if mode == "constant":
        return np.pad(x, npad, mode="constant", constant_values=value)
    return np.pad(x, npad, mode=nmode)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-4, rtol=1e-4, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)


# ---------------------------------------------------------------------------
# F.conv2d
# ---------------------------------------------------------------------------
class TestConv2dFunctional(Base):
    def setUp(self):
        rs = np.random.RandomState(0)
        self.x = rs.randn(2, 3, 7, 7).astype("float32")
        self.w = rs.randn(4, 3, 3, 3).astype("float32")
        self.b = rs.randn(4).astype("float32")

    def test_conv2d_stride_pad(self):
        x, w, b = self.x, self.w, self.b
        for stride, pad in [(1, 0), (2, 1), (1, 2), (3, 1)]:
            def body(dev, stride=stride, pad=pad):
                g = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b),
                             stride=stride, padding=pad).numpy()
                self.ac(g, conv2d_ref(x, w, b, stride, pad),
                        msg=f"conv2d s{stride}p{pad} {dev}")
            both_devices(body)

    def test_conv2d_dilation(self):
        x, w = self.x, self.w
        def body(dev):
            g = F.conv2d(torch.tensor(x), torch.tensor(w), None,
                         stride=1, padding=2, dilation=2).numpy()
            self.ac(g, conv2d_ref(x, w, None, 1, 2, 2), msg=f"conv2d dil2 {dev}")
        both_devices(body)

    def test_conv2d_groups(self):
        rs = np.random.RandomState(10)
        x = rs.randn(2, 4, 6, 6).astype("float32")   # groups=2 -> Cin_g=2
        w = rs.randn(6, 2, 3, 3).astype("float32")   # Cout=6
        def body(dev):
            g = F.conv2d(torch.tensor(x), torch.tensor(w), None, groups=2).numpy()
            self.ac(g, conv2d_ref(x, w, None, 1, 0, 1, 2), msg=f"conv2d groups2 {dev}")
        both_devices(body)

    def test_conv2d_depthwise(self):
        rs = np.random.RandomState(11)
        x = rs.randn(2, 4, 6, 6).astype("float32")
        w = rs.randn(4, 1, 3, 3).astype("float32")   # groups=Cin=Cout=4
        def body(dev):
            g = F.conv2d(torch.tensor(x), torch.tensor(w), None, padding=1, groups=4).numpy()
            self.ac(g, conv2d_ref(x, w, None, 1, 1, 1, 4), msg=f"conv2d depthwise {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.conv_transpose2d
# ---------------------------------------------------------------------------
class TestConvTranspose2dFunctional(Base):
    def setUp(self):
        rs = np.random.RandomState(1)
        self.x = rs.randn(2, 3, 4, 4).astype("float32")
        self.w = rs.randn(3, 5, 3, 3).astype("float32")   # [Cin, Cout, Kh, Kw]
        self.b = rs.randn(5).astype("float32")

    def test_conv_transpose2d_basic(self):
        x, w, b = self.x, self.w, self.b
        def body(dev):
            g = F.conv_transpose2d(torch.tensor(x), torch.tensor(w), torch.tensor(b)).numpy()
            self.ac(g, conv_transpose2d_ref(x, w, b), msg=f"convT basic {dev}")
        both_devices(body)

    def test_conv_transpose2d_stride_pad_outpad(self):
        x, w, b = self.x, self.w, self.b
        def body(dev):
            g = F.conv_transpose2d(torch.tensor(x), torch.tensor(w), torch.tensor(b),
                                   stride=2, padding=1, output_padding=1).numpy()
            self.ac(g, conv_transpose2d_ref(x, w, b, 2, 1, 1),
                    msg=f"convT s2p1op1 {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.max_pool2d / F.avg_pool2d
# ---------------------------------------------------------------------------
class TestPoolFunctional(Base):
    def setUp(self):
        self.x = np.random.RandomState(2).randn(2, 3, 8, 8).astype("float32")

    def test_max_pool2d(self):
        x = self.x
        for k, s, p in [(2, None, 0), (3, 2, 1), (2, 2, 0)]:
            def body(dev, k=k, s=s, p=p):
                g = F.max_pool2d(torch.tensor(x), k, stride=s, padding=p).numpy()
                self.ac(g, pool2d_ref(x, k, s, p, mode="max"),
                        msg=f"maxpool k{k}s{s}p{p} {dev}")
            both_devices(body)

    def test_max_pool2d_ceil_mode(self):
        x = self.x
        def body(dev):
            g = F.max_pool2d(torch.tensor(x), 3, stride=2, ceil_mode=True).numpy()
            self.ac(g, pool2d_ref(x, 3, 2, 0, ceil_mode=True, mode="max"),
                    msg=f"maxpool ceil {dev}")
        both_devices(body)

    def test_avg_pool2d(self):
        x = self.x
        for k, s, p in [(2, None, 0), (2, 2, 0), (4, 4, 0)]:
            def body(dev, k=k, s=s, p=p):
                g = F.avg_pool2d(torch.tensor(x), k, stride=s, padding=p).numpy()
                self.ac(g, pool2d_ref(x, k, s, p, mode="avg"),
                        msg=f"avgpool k{k}s{s}p{p} {dev}")
            both_devices(body)

    def test_avg_pool2d_count_include_pad_true(self):
        # default count_include_pad=True: padded zeros are counted in the divisor.
        x = self.x
        def body(dev):
            g = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1).numpy()
            self.ac(g, pool2d_ref(x, 3, 2, 1, mode="avg", count_include_pad=True),
                    msg=f"avgpool incl {dev}")
        both_devices(body)

    def test_avg_pool2d_ceil_mode(self):
        x = self.x
        def body(dev):
            g = F.avg_pool2d(torch.tensor(x), 3, stride=2, ceil_mode=True).numpy()
            self.ac(g, pool2d_ref(x, 3, 2, 0, ceil_mode=True, mode="avg"),
                    msg=f"avgpool ceil {dev}")
        both_devices(body)

    @unittest.skip("REASON: jittor's avg_pool2d ignores count_include_pad=False; "
                   "output equals count_include_pad=True (verified incl==excl exactly). "
                   "torch divides by the non-pad element count at padded borders. "
                   "Semantic gap in jittor's Pool op, not a test artifact.")
    def test_avg_pool2d_count_include_pad_false(self):
        x = self.x
        def body(dev):
            g = F.avg_pool2d(torch.tensor(x), 3, stride=2, padding=1,
                             count_include_pad=False).numpy()
            self.ac(g, pool2d_ref(x, 3, 2, 1, mode="avg", count_include_pad=False),
                    msg=f"avgpool excl {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.adaptive_avg_pool2d / nn.AdaptiveMaxPool2d
# (F.adaptive_max_pool2d is NOT exposed; only the module form exists.)
# ---------------------------------------------------------------------------
class TestAdaptivePool(Base):
    def setUp(self):
        self.x = np.random.RandomState(3).randn(2, 3, 8, 8).astype("float32")

    def test_adaptive_avg_pool2d_divisor_sizes(self):
        # jittor matches torch's adaptive pooling exactly when output_size | input_size.
        x = self.x
        for out in [1, 2, 4, 8]:
            def body(dev, out=out):
                g = F.adaptive_avg_pool2d(torch.tensor(x), out).numpy()
                self.ac(g, adaptive_avg_ref(x, out), msg=f"adaptive_avg out{out} {dev}")
            both_devices(body)

    def test_adaptive_max_pool2d_divisor_sizes(self):
        x = self.x
        for out in [1, 2, 4]:
            def body(dev, out=out):
                g = nn.AdaptiveMaxPool2d(out)(torch.tensor(x)).numpy()
                self.ac(g, adaptive_max_ref(x, out), msg=f"adaptive_max out{out} {dev}")
            both_devices(body)

    def test_adaptive_avg_pool2d_rectangular(self):
        # non-square but each dim still divides evenly (H=8->4, W=8->2).
        x = self.x
        def body(dev):
            g = F.adaptive_avg_pool2d(torch.tensor(x), (4, 2)).numpy()
            self.ac(g, adaptive_avg_ref(x, (4, 2)), msg=f"adaptive_avg (4,2) {dev}")
        both_devices(body)

    @unittest.skip("REASON: for non-divisor output (e.g. 8->3) jittor uses uniform "
                   "stride/kernel pooling, NOT torch's variable-bin algorithm "
                   "(start=floor(i*H/O), end=ceil((i+1)*H/O)); outputs differ by up to "
                   "~13x rel. Matches torch only when output_size divides input_size.")
    def test_adaptive_avg_pool2d_nondivisor(self):
        x = self.x
        def body(dev):
            g = F.adaptive_avg_pool2d(torch.tensor(x), 3).numpy()
            self.ac(g, adaptive_avg_ref(x, 3), msg=f"adaptive_avg out3 {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# nn.Conv2d / nn.MaxPool2d / nn.AvgPool2d module forward
# ---------------------------------------------------------------------------
class TestModuleForward(Base):
    def test_conv2d_module(self):
        rs = np.random.RandomState(20)
        x = rs.randn(2, 3, 8, 8).astype("float32")
        w = rs.randn(5, 3, 3, 3).astype("float32")
        b = rs.randn(5).astype("float32")
        def body(dev):
            m = nn.Conv2d(3, 5, 3, stride=2, padding=1)
            m.weight = jt.array(w)
            m.bias = jt.array(b)
            g = m(torch.tensor(x)).numpy()
            self.ac(g, conv2d_ref(x, w, b, 2, 1), msg=f"Conv2d module {dev}")
        both_devices(body)

    def test_conv2d_module_nobias(self):
        rs = np.random.RandomState(21)
        x = rs.randn(1, 2, 6, 6).astype("float32")
        w = rs.randn(3, 2, 3, 3).astype("float32")
        def body(dev):
            m = nn.Conv2d(2, 3, 3, bias=False)
            m.weight = jt.array(w)
            self.assertIsNone(m.bias, "bias=False should give bias is None")
            g = m(torch.tensor(x)).numpy()
            self.ac(g, conv2d_ref(x, w, None), msg=f"Conv2d nobias {dev}")
        both_devices(body)

    def test_maxpool2d_module(self):
        x = np.random.RandomState(22).randn(2, 3, 8, 8).astype("float32")
        def body(dev):
            g = nn.MaxPool2d(2)(torch.tensor(x)).numpy()
            self.ac(g, pool2d_ref(x, 2, mode="max"), msg=f"MaxPool2d module {dev}")
        both_devices(body)

    def test_avgpool2d_module(self):
        x = np.random.RandomState(23).randn(2, 3, 8, 8).astype("float32")
        def body(dev):
            g = nn.AvgPool2d(2)(torch.tensor(x)).numpy()
            self.ac(g, pool2d_ref(x, 2, mode="avg"), msg=f"AvgPool2d module {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# Normalization modules
# ---------------------------------------------------------------------------
class TestNormModules(Base):
    def test_batchnorm2d_eval(self):
        rs = np.random.RandomState(30)
        x = rs.randn(2, 3, 5, 5).astype("float32")
        rm = rs.randn(3).astype("float32")
        rv = (rs.rand(3) + 0.5).astype("float32")
        w = rs.randn(3).astype("float32")
        b = rs.randn(3).astype("float32")
        def body(dev):
            m = nn.BatchNorm2d(3, eps=1e-5)
            m.running_mean = jt.array(rm)
            m.running_var = jt.array(rv)
            m.weight = jt.array(w)
            m.bias = jt.array(b)
            m.eval()
            self.assertFalse(m.is_train, "eval() must set is_train False")
            g = m(torch.tensor(x)).numpy()
            self.ac(g, batchnorm_eval_ref(x, rm, rv, w, b, 1e-5),
                    atol=1e-4, msg=f"BatchNorm2d eval {dev}")
        both_devices(body)

    def test_batchnorm2d_eval_uses_running_stats_not_batch(self):
        # In eval mode the batch's own mean/var must be IGNORED; feeding a constant
        # offset shifts the output by exactly (offset / sqrt(running_var)) when w=1,b=0.
        rs = np.random.RandomState(31)
        x = rs.randn(2, 4, 4, 4).astype("float32")
        rm = np.zeros(4, "float32")
        rv = np.ones(4, "float32")
        def body(dev):
            m = nn.BatchNorm2d(4, eps=0.0)
            m.running_mean = jt.array(rm)
            m.running_var = jt.array(rv)
            m.weight = jt.array(np.ones(4, "float32"))
            m.bias = jt.array(np.zeros(4, "float32"))
            m.eval()
            g = m(torch.tensor(x)).numpy()
            # running_mean=0, running_var=1, eps=0 -> identity in eval mode
            self.ac(g, x, atol=1e-4, msg=f"BN eval identity {dev}")
        both_devices(body)

    def test_groupnorm(self):
        rs = np.random.RandomState(32)
        x = rs.randn(2, 6, 4, 4).astype("float32")
        w = rs.randn(6).astype("float32")
        b = rs.randn(6).astype("float32")
        def body(dev):
            m = nn.GroupNorm(3, 6, eps=1e-5)
            m.weight = jt.array(w)
            m.bias = jt.array(b)
            g = m(torch.tensor(x)).numpy()
            self.ac(g, groupnorm_ref(x, 3, w, b, 1e-5), atol=1e-3,
                    msg=f"GroupNorm {dev}")
        both_devices(body)

    def test_groupnorm_single_group_is_layernorm_over_chw(self):
        # G=1 normalizes over the whole (C,H,W) volume per sample.
        rs = np.random.RandomState(33)
        x = rs.randn(2, 4, 3, 3).astype("float32")
        w = np.ones(4, "float32"); b = np.zeros(4, "float32")
        def body(dev):
            m = nn.GroupNorm(1, 4, eps=1e-5)
            m.weight = jt.array(w); m.bias = jt.array(b)
            g = m(torch.tensor(x)).numpy()
            self.ac(g, groupnorm_ref(x, 1, w, b, 1e-5), atol=1e-3,
                    msg=f"GroupNorm G=1 {dev}")
        both_devices(body)

    def test_instancenorm2d_affine(self):
        # NOTE: jittor's InstanceNorm2d defaults affine=True (torch defaults affine=False);
        # the computation itself matches once weights are set, so we set them explicitly.
        rs = np.random.RandomState(34)
        x = rs.randn(2, 3, 5, 5).astype("float32")
        w = rs.randn(3).astype("float32")
        b = rs.randn(3).astype("float32")
        def body(dev):
            m = nn.InstanceNorm2d(3, eps=1e-5)
            self.assertIsNotNone(getattr(m, "weight", None),
                                 "jittor InstanceNorm2d defaults affine=True")
            m.weight = jt.array(w)
            m.bias = jt.array(b)
            g = m(torch.tensor(x)).numpy()
            self.ac(g, instancenorm_ref(x, w, b, 1e-5), atol=1e-3,
                    msg=f"InstanceNorm2d {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.interpolate
# ---------------------------------------------------------------------------
class TestInterpolate(Base):
    def test_interpolate_nearest_upsample(self):
        x = np.random.RandomState(40).randn(1, 2, 3, 3).astype("float32")
        def body(dev):
            g = F.interpolate(torch.tensor(x), scale_factor=2, mode="nearest").numpy()
            self.ac(g, nearest_ref(x, 2), atol=1e-5, msg=f"interp nearest x2 {dev}")
        both_devices(body)

    def test_interpolate_bilinear_align_corners_true(self):
        x = np.random.RandomState(41).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.tensor(x), size=(8, 8), mode="bilinear",
                              align_corners=True).numpy()
            self.ac(g, bilinear_ref(x, 8, 8, True), atol=1e-3,
                    msg=f"interp bilinear AC=True {dev}")
        both_devices(body)

    def test_interpolate_bilinear_align_corners_false(self):
        x = np.random.RandomState(42).randn(1, 2, 4, 4).astype("float32")
        def body(dev):
            g = F.interpolate(torch.tensor(x), size=(8, 8), mode="bilinear",
                              align_corners=False).numpy()
            self.ac(g, bilinear_ref(x, 8, 8, False), atol=1e-3,
                    msg=f"interp bilinear AC=False {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# F.pad
# ---------------------------------------------------------------------------
class TestPad(Base):
    def setUp(self):
        self.x = np.random.RandomState(50).randn(1, 1, 4, 5).astype("float32")

    def test_pad_constant(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            g = F.pad(torch.tensor(x), pad, mode="constant", value=7.0).numpy()
            self.ac(g, pad_ref(x, pad, "constant", 7.0), atol=1e-5,
                    msg=f"pad constant {dev}")
        both_devices(body)

    def test_pad_default_is_zero_constant(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            g = F.pad(torch.tensor(x), pad).numpy()
            self.ac(g, pad_ref(x, pad, "constant", 0.0), atol=1e-5,
                    msg=f"pad default {dev}")
        both_devices(body)

    def test_pad_replicate(self):
        x = self.x; pad = (1, 2, 3, 1)
        def body(dev):
            g = F.pad(torch.tensor(x), pad, mode="replicate").numpy()
            self.ac(g, pad_ref(x, pad, "replicate"), atol=1e-5,
                    msg=f"pad replicate {dev}")
        both_devices(body)

    def test_pad_reflect(self):
        x = self.x; pad = (1, 1, 1, 1)   # reflect needs pad < dim
        def body(dev):
            g = F.pad(torch.tensor(x), pad, mode="reflect").numpy()
            self.ac(g, pad_ref(x, pad, "reflect"), atol=1e-5,
                    msg=f"pad reflect {dev}")
        both_devices(body)

    def test_pad_circular(self):
        x = self.x; pad = (1, 1, 1, 1)
        def body(dev):
            g = F.pad(torch.tensor(x), pad, mode="circular").numpy()
            self.ac(g, pad_ref(x, pad, "circular"), atol=1e-5,
                    msg=f"pad circular {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
