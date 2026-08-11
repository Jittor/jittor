# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Conv/pool OpInfos: conv2d, conv_transpose2d, max_pool2d, avg_pool2d, adaptive_avg_pool2d.

The convolution/pooling domain of the registry. The numpy reference forwards
(``conv2d_ref``, ``conv_transpose2d_ref``, ``pool2d_ref``, ``adaptive_avg_ref``) are
adapted faithfully from the validated, bit-faithful-to-PyTorch implementations in
``test_torch_compat_conv_pool.py`` -- they are a must-preserve asset, so they are
reproduced here (lightly rewrapped so each op's reference kwargs match the jittor
callable's kwargs exactly).

Differentiation contract (see ``test_ops.py._diff_plan``): a SampleInput's ``input``
plus any *positional* float Vars are gradchecked; everything in ``kwargs`` is forwarded
verbatim to both the op and the numpy ref and is NOT differentiated. So input / weight /
bias are passed positionally (all float, all gradchecked), while stride / padding /
dilation / groups / kernel_size / count_include_pad / output_size are passed as kwargs
(or, for ``output_size``, as a non-Var positional that the diff-plan skips).

Tensors are kept small (<= ~24 differentiated elements) because gradcheck is O(numel)
forward passes.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# --------------------------------------------------------------------- helpers

def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


# ------------------------------------------------------------------ numpy refs
# Adapted verbatim from the validated references in test_torch_compat_conv_pool.py.
# Each ref takes the *same* keyword names as the jittor op it mirrors so the generic
# driver can forward ``**sample.kwargs`` to both sides unchanged.

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


# -- thin ref wrappers whose kwarg names match the jittor op kwargs exactly ----

def max_pool2d_ref(x, kernel_size, stride=None, padding=0, ceil_mode=False):
    return pool2d_ref(x, kernel_size, stride=stride, padding=padding,
                      ceil_mode=ceil_mode, mode="max")


def avg_pool2d_ref(x, kernel_size, stride=None, padding=0, ceil_mode=False,
                   count_include_pad=True):
    return pool2d_ref(x, kernel_size, stride=stride, padding=padding,
                      ceil_mode=ceil_mode, mode="avg",
                      count_include_pad=count_include_pad)


# --------------------------------------------------------------- sample builders
# input / weight / bias are positional float Vars => all gradchecked.
# every hyper-parameter is a kwarg (forwarded to op and ref, never differentiated).

def sample_conv2d(op_info, device, dtype, requires_grad):
    # (N, Cin, H, W), (Cout, Cin/groups, Kh, Kw), bias?, kwargs
    cases = [
        # plain stride/pad
        ((1, 2, 5, 5), (3, 2, 3, 3), True, dict(stride=1, padding=0)),
        ((1, 2, 5, 5), (2, 2, 3, 3), True, dict(stride=2, padding=1)),
        # rectangular stride/padding tuples, no bias
        ((1, 2, 5, 4), (2, 2, 3, 3), False, dict(stride=(2, 1), padding=(1, 0))),
        # dilation
        ((1, 1, 5, 5), (2, 1, 2, 2), False, dict(stride=1, padding=1, dilation=2)),
        # grouped conv: Cin=4, groups=2 -> Cin/groups=2
        ((1, 4, 4, 4), (4, 2, 3, 3), False, dict(padding=1, groups=2)),
        # depthwise: groups=Cin=Cout
        ((1, 3, 4, 4), (3, 1, 3, 3), False, dict(padding=1, groups=3)),
    ]
    out = []
    for i, (xs, ws, has_bias, kw) in enumerate(cases):
        x = make_tensor(*xs, dtype=dtype, requires_grad=requires_grad, seed=600 + i)
        w = make_tensor(*ws, dtype=dtype, requires_grad=requires_grad, seed=620 + i)
        if has_bias:
            b = make_tensor(ws[0], dtype=dtype, requires_grad=requires_grad, seed=640 + i)
            out.append(SampleInput(x, w, b, **kw))
        else:
            out.append(SampleInput(x, w, **kw))
    return out


def sample_conv_transpose2d(op_info, device, dtype, requires_grad):
    # weight layout is (Cin, Cout/groups, Kh, Kw).
    # groups=1 only: the conv_transpose grouped numpy ref below is reproduced from
    # the validated test, but only its groups=1 path was checked bit-faithful to
    # torch there, so grouped transpose is intentionally not sampled.
    cases = [
        ((1, 2, 4, 4), (2, 3, 3, 3), True, dict(stride=1, padding=0)),
        ((1, 2, 4, 4), (2, 2, 3, 3), True, dict(stride=2, padding=1, output_padding=1)),
        ((1, 2, 4, 3), (2, 2, 3, 3), False, dict(stride=(2, 1), padding=(1, 0))),
        ((1, 1, 4, 4), (1, 2, 2, 2), False, dict(stride=1, padding=1, dilation=2)),
    ]
    out = []
    for i, (xs, ws, has_bias, kw) in enumerate(cases):
        x = make_tensor(*xs, dtype=dtype, requires_grad=requires_grad, seed=660 + i)
        w = make_tensor(*ws, dtype=dtype, requires_grad=requires_grad, seed=680 + i)
        if has_bias:
            cout = ws[1] * kw.get("groups", 1)
            b = make_tensor(cout, dtype=dtype, requires_grad=requires_grad, seed=700 + i)
            out.append(SampleInput(x, w, b, **kw))
        else:
            out.append(SampleInput(x, w, **kw))
    return out


def sample_max_pool2d(op_info, device, dtype, requires_grad):
    # Generic (non-integer, well-separated) values keep the argmax unique, so the
    # subgradient at ties is not exercised and gradcheck of the chosen-element
    # gradient is well defined.
    cases = [
        dict(kernel_size=2, stride=2, padding=0),
        dict(kernel_size=2, stride=1, padding=0),
        dict(kernel_size=3, stride=2, padding=1),
        dict(kernel_size=2, stride=2, padding=0, ceil_mode=True),
    ]
    out = []
    for i, kw in enumerate(cases):
        x = make_tensor(1, 2, 5, 5, dtype=dtype, low=-4.0, high=4.0,
                        requires_grad=requires_grad, seed=720 + i)
        out.append(SampleInput(x, **kw))
    return out


def sample_avg_pool2d(op_info, device, dtype, requires_grad):
    cases = [
        dict(kernel_size=2, stride=2, padding=0),
        dict(kernel_size=2, stride=1, padding=0),
        dict(kernel_size=3, stride=2, padding=1, count_include_pad=True),
        dict(kernel_size=3, stride=2, padding=1, count_include_pad=False),
        dict(kernel_size=2, stride=2, padding=0, ceil_mode=True),
    ]
    out = []
    for i, kw in enumerate(cases):
        x = make_tensor(1, 2, 5, 5, dtype=dtype, requires_grad=requires_grad, seed=740 + i)
        out.append(SampleInput(x, **kw))
    return out


def sample_adaptive_avg_pool2d(op_info, device, dtype, requires_grad):
    # output_size is a non-Var positional => forwarded to ref, not differentiated.
    out = []
    for i, osz in enumerate([1, 2, (2, 4), (4, 2)]):
        x = make_tensor(1, 2, 4, 4, dtype=dtype, requires_grad=requires_grad, seed=760 + i)
        out.append(SampleInput(x, osz))
    return out


op_db = [
    # ---- convolutions: input + weight + bias all differentiated ----
    # reference_tol: conv sums Ci*Kh*Kw products of O(1) values, so the float32 forward
    # accumulates more round-off than the per-dtype default atol allows (a single element
    # can miss 1e-5 by ~2x on outputs of magnitude ~10). 1e-4 is still ~1e-6 relative --
    # far tighter than any real kernel bug. (gradcheck runs in float64, unaffected.)
    OpInfo("conv2d", op=nn.conv2d, ref=conv2d_ref,
           sample_inputs_func=sample_conv2d, reference_tol=(1e-4, 1e-4)),
    OpInfo("conv_transpose2d", op=nn.conv_transpose2d, ref=conv_transpose2d_ref,
           sample_inputs_func=sample_conv_transpose2d, reference_tol=(1e-4, 1e-4)),

    # ---- pooling ----
    # max_pool2d backward is a subgradient (gather of the argmax element); with
    # well-separated random inputs the argmax is unique, so gradcheck of the chosen
    # gradient is well defined. gradgrad is exactly zero (piecewise-linear) -> keep
    # the 2nd-derivative check off to avoid a vacuous gradgradcheck pass/fragility.
    OpInfo("max_pool2d", op=nn.max_pool2d, ref=max_pool2d_ref,
           sample_inputs_func=sample_max_pool2d, supports_gradgrad=False),
    OpInfo("avg_pool2d", op=nn.avg_pool2d, ref=avg_pool2d_ref,
           sample_inputs_func=sample_avg_pool2d),
    OpInfo("adaptive_avg_pool2d", op=nn.adaptive_avg_pool2d, ref=adaptive_avg_ref,
           sample_inputs_func=sample_adaptive_avg_pool2d),
]
