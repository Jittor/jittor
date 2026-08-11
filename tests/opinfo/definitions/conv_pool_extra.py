# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""1-D / 3-D convolution and pooling OpInfos -- the conv/pool variants beyond the
``conv2d`` / ``max_pool2d`` / ``avg_pool2d`` already in ``nn_conv_pool.py``.

``conv2d`` is the workhorse and was covered first; the 1-D and 3-D kernels are SEPARATE
codegen paths (different reindex loops) and a bug in one is invisible to the other. Same
for pooling. Each here gets an INDEPENDENT numpy reference (explicit cross-correlation /
windowed max-mean), a gradchecked backward, and CPU-vs-CUDA parity.

References are written for the exact regime the samples use (groups=1, dilation=1, and a
small set of stride/padding values) so the numpy oracle is trivially auditable rather than
a second full conv implementation. ``conv1d``/``conv3d`` differentiate input + weight (+
bias when present); ``max_pool`` is piecewise so its gradgrad is off (like the 2-D one).
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo


# ----------------------------------------------------------------- conv numpy refs
def conv1d_ref(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    N, C, L = x.shape
    Co, Ci, K = weight.shape
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    Lp = x.shape[2]
    Lo = (Lp - K) // stride + 1
    out = np.zeros((N, Co, Lo), dtype=np.float64)
    for i in range(Lo):
        seg = x[:, :, i * stride:i * stride + K]            # N, Ci, K
        out[:, :, i] = np.einsum("nck,ock->no", seg, weight)
    if bias is not None:
        out = out + np.asarray(bias)[None, :, None]
    return out


def conv3d_ref(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    N, C, D, H, W = x.shape
    Co, Ci, kd, kh, kw = weight.shape
    if padding:
        p = padding
        x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p), (p, p)))
    Dp, Hp, Wp = x.shape[2:]
    Do = (Dp - kd) // stride + 1
    Ho = (Hp - kh) // stride + 1
    Wo = (Wp - kw) // stride + 1
    out = np.zeros((N, Co, Do, Ho, Wo), dtype=np.float64)
    for a in range(Do):
        for b in range(Ho):
            for c in range(Wo):
                seg = x[:, :, a * stride:a * stride + kd,
                        b * stride:b * stride + kh,
                        c * stride:c * stride + kw]          # N, Ci, kd, kh, kw
                out[:, :, a, b, c] = np.einsum("ncdhw,ocdhw->no", seg, weight)
    if bias is not None:
        out = out + np.asarray(bias)[None, :, None, None, None]
    return out


# ----------------------------------------------------------------- pooling numpy refs
def maxpool1d_ref(x, k=2):
    N, C, L = x.shape
    Lo = L // k
    return x[:, :, :Lo * k].reshape(N, C, Lo, k).max(-1)


def avgpool1d_ref(x, k=2):
    N, C, L = x.shape
    Lo = L // k
    return x[:, :, :Lo * k].reshape(N, C, Lo, k).mean(-1)


def maxpool3d_ref(x, k=2):
    N, C, D, H, W = x.shape
    D2, H2, W2 = D // k * k, H // k * k, W // k * k
    return x[:, :, :D2, :H2, :W2].reshape(
        N, C, D // k, k, H // k, k, W // k, k).max(axis=(3, 5, 7))


def avgpool3d_ref(x, k=2):
    N, C, D, H, W = x.shape
    D2, H2, W2 = D // k * k, H // k * k, W // k * k
    return x[:, :, :D2, :H2, :W2].reshape(
        N, C, D // k, k, H // k, k, W // k, k).mean(axis=(3, 5, 7))


# ----------------------------------------------------------------- pooling op wrappers
def _maxpool1d(x):  return nn.MaxPool1d(2)(x)
def _avgpool1d(x):  return nn.AvgPool1d(2)(x)
def _maxpool3d(x):  return nn.MaxPool3d(2)(x)
def _avgpool3d(x):  return nn.AvgPool3d(2)(x)


# --------------------------------------------------------------- sample builders
def sample_conv1d(op_info, device, dtype, requires_grad):
    out = []
    # (stride, padding, with_bias) regimes; input/weight kept tiny (gradcheck is O(numel)).
    specs = [(1, 0, False), (2, 0, False), (1, 1, True)]
    for i, (stride, padding, has_bias) in enumerate(specs):
        x = make_tensor(1, 2, 6, dtype=dtype, requires_grad=requires_grad, seed=1700 + i)
        w = make_tensor(2, 2, 3, dtype=dtype, requires_grad=requires_grad, seed=1710 + i)
        if has_bias:
            b = make_tensor(2, dtype=dtype, requires_grad=requires_grad, seed=1720 + i)
            out.append(SampleInput(x, w, b, stride=stride, padding=padding))
        else:
            out.append(SampleInput(x, w, stride=stride, padding=padding))
    return out


def sample_conv3d(op_info, device, dtype, requires_grad):
    out = []
    specs = [(1, 0, False), (1, 1, True)]
    for i, (stride, padding, has_bias) in enumerate(specs):
        x = make_tensor(1, 1, 3, 3, 3, dtype=dtype, requires_grad=requires_grad, seed=1730 + i)
        w = make_tensor(1, 1, 2, 2, 2, dtype=dtype, requires_grad=requires_grad, seed=1740 + i)
        if has_bias:
            b = make_tensor(1, dtype=dtype, requires_grad=requires_grad, seed=1750 + i)
            out.append(SampleInput(x, w, b, stride=stride, padding=padding))
        else:
            out.append(SampleInput(x, w, stride=stride, padding=padding))
    return out


def sample_pool1d(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(2, 3, 8, dtype=dtype, requires_grad=requires_grad,
                                    seed=1760 + i)) for i in range(2)]


def sample_pool3d(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(1, 2, 4, 4, 4, dtype=dtype, requires_grad=requires_grad,
                                    seed=1770 + i)) for i in range(2)]


op_db = [
    # ---- convolutions: input + weight (+ bias) differentiated ----
    OpInfo("conv1d", op=nn.conv1d, ref=conv1d_ref, sample_inputs_func=sample_conv1d),
    OpInfo("conv3d", op=nn.conv3d, ref=conv3d_ref, sample_inputs_func=sample_conv3d),

    # ---- pooling: avg smooth (gradgrad on), max piecewise (gradgrad off) ----
    OpInfo("max_pool1d", op=_maxpool1d, ref=maxpool1d_ref,
           sample_inputs_func=sample_pool1d, supports_gradgrad=False),
    OpInfo("avg_pool1d", op=_avgpool1d, ref=avgpool1d_ref,
           sample_inputs_func=sample_pool1d),
    OpInfo("max_pool3d", op=_maxpool3d, ref=maxpool3d_ref,
           sample_inputs_func=sample_pool3d, supports_gradgrad=False),
    OpInfo("avg_pool3d", op=_avgpool3d, ref=avgpool3d_ref,
           sample_inputs_func=sample_pool3d),
]
