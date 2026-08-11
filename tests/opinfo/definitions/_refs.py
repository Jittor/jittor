# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Shared numpy references and sample-input builders for OpInfo definitions.

Every definition module imports from here so the numpy reference implementations
(the INDEPENDENT forward oracle) and the input generators are written once and stay
consistent. Import surface for a definition module is typically::

    from .._refs import *           # make_tensor, SampleInput, np, the refs
    from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo
"""
import math

import numpy as np
import jittor as jt
from jittor import nn

from _helpers import common as cu
from ..core import SampleInput

F = nn.functional

__all__ = [
    "math", "np", "jt", "nn", "F", "cu", "SampleInput", "make_tensor",
    "sigmoid_ref", "erf_ref", "gelu_ref", "silu_ref", "softmax_ref",
    "log_softmax_ref", "reduce_ref", "layer_norm_ref", "group_norm_ref",
    "sample_unary", "sample_binary", "sample_reduction",
]

make_tensor = cu.make_tensor


# ------------------------------------------------------------------- numpy refs

def sigmoid_ref(x):
    return 1.0 / (1.0 + np.exp(-x))


erf_ref = np.vectorize(math.erf)


def gelu_ref(x):
    return 0.5 * x * (1.0 + erf_ref(x / math.sqrt(2.0)))


def silu_ref(x):
    return x * sigmoid_ref(x)


def softmax_ref(x, dim=-1):
    e = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e / np.sum(e, axis=dim, keepdims=True)


def log_softmax_ref(x, dim=-1):
    m = np.max(x, axis=dim, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=dim, keepdims=True))


def reduce_ref(npfn):
    """Wrap a numpy reduction so it matches jittor's (dim, keepdims) kwargs and
    its (1,)-shaped full-reduce result (jittor has no 0-d scalar)."""
    def ref(x, dim=None, keepdims=False):
        return np.atleast_1d(npfn(x, axis=dim, keepdims=keepdims))
    return ref


def layer_norm_ref(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    ndims = len(normalized_shape)
    axes = tuple(range(x.ndim - ndims, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def group_norm_ref(x, num_groups, weight=None, bias=None, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    xr = x.reshape(N, num_groups, C // num_groups, *spatial)
    axes = tuple(range(2, xr.ndim))
    mean = xr.mean(axis=axes, keepdims=True)
    var = xr.var(axis=axes, keepdims=True)
    xr = (xr - mean) / np.sqrt(var + eps)
    out = xr.reshape(x.shape)
    if weight is not None:
        out = out * weight.reshape((1, C) + (1,) * len(spatial))
    if bias is not None:
        out = out + bias.reshape((1, C) + (1,) * len(spatial))
    return out


# --------------------------------------------------------------- sample builders

def sample_unary(op_info, device, dtype, requires_grad):
    lo, hi = getattr(op_info, "domain", (None, None))
    shapes = [(5,), (3, 4), (2, 3, 4)]
    return [SampleInput(make_tensor(*s, dtype=dtype, low=lo, high=hi,
                                    requires_grad=requires_grad, seed=100 + i))
            for i, s in enumerate(shapes)]


def sample_binary(op_info, device, dtype, requires_grad):
    pairs = [((3, 4), (3, 4)), ((3, 4), (4,)), ((2, 1, 4), (3, 4))]   # incl. broadcast
    out = []
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, requires_grad=requires_grad, seed=200 + i)
        b = make_tensor(*sb, dtype=dtype, requires_grad=requires_grad, seed=250 + i,
                        low=0.5, high=2.0)   # keep >0 so div/pow refs stay finite
        out.append(SampleInput(a, b))
    return out


def sample_reduction(op_info, device, dtype, requires_grad):
    out = [SampleInput(make_tensor(3, 4, 5, dtype=dtype, requires_grad=requires_grad,
                                   seed=300))]
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdims in (False, True):
            out.append(SampleInput(
                make_tensor(3, 4, 5, dtype=dtype, requires_grad=requires_grad, seed=310 + i),
                dim=dim, keepdims=keepdims))
    return out
