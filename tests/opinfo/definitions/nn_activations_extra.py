# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""More jittor-core nn functional ops (``jittor.nn``, ``jittor.misc``) that had no
op_db entry: ``hardtanh``, ``glu``, ``log_sigmoid``, ``normalize``, ``cosine_similarity``.

All are DIFFERENTIABLE and were classic forward-only suspects -- the value here is the
gradchecked backward (and the CPU-vs-CUDA parity it then rides for free):

  * ``hardtanh``  -- clamp activation; piecewise, so gradgrad differentiates the step mask
    (degenerate) -> gradgrad off, like relu6/hardswish.
  * ``glu``       -- gated linear unit ``a * sigmoid(b)`` over a halved axis; a backward
    that mis-pairs the two halves is invisible to a forward check.
  * ``log_sigmoid`` -- ``log(sigmoid(x))``; the numerically-stable form's gradient is a
    plain ``sigmoid(-x)`` that is easy to stub with the wrong sign.
  * ``normalize`` -- L2 normalize; the quotient rule makes the gradient couple every
    component (a per-element backward is wrong) -- exactly the silent-wrong shape.
  * ``cosine_similarity`` -- two-operand, gradient w.r.t. BOTH inputs.

Each has an INDEPENDENT numpy reference. Samples keep vectors away from the zero-norm
singularity (normalize / cosine) so the backward is smooth over the gradcheck step.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo, UnaryUfuncInfo


# ----------------------------------------------------------------- numpy refs
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def hardtanh_ref(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)


def log_sigmoid_ref(x):
    return np.log(_sig(x))


def glu_ref(x, dim=-1):
    half = x.shape[dim] // 2
    a = np.take(x, range(half), axis=dim)
    b = np.take(x, range(half, x.shape[dim]), axis=dim)
    return a * _sig(b)


def normalize_ref(x, p=2, dim=1, eps=1e-12):
    n = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_similarity_ref(x1, x2, dim=1, eps=1e-8):
    num = (x1 * x2).sum(axis=dim)
    den = np.maximum(np.linalg.norm(x1, axis=dim) * np.linalg.norm(x2, axis=dim), eps)
    return num / den


# --------------------------------------------------------------- sample builders
def sample_hardtanh(op_info, device, dtype, requires_grad):
    # values straddle the +/-1 knees so the clamp activates on both sides; keep samples
    # off the exact knee (gradcheck would straddle the kink there).
    out = []
    for i, s in enumerate([(5,), (3, 4), (2, 3, 4)]):
        x = make_tensor(*s, dtype=dtype, low=-2.5, high=2.5,
                        requires_grad=requires_grad, seed=1500 + i)
        out.append(SampleInput(x))
    return out


def sample_glu(op_info, device, dtype, requires_grad):
    # the glu axis must be EVEN (it is split in half); sweep dim.
    out = []
    specs = [((2, 4), -1), ((3, 6), -1), ((2, 4, 3), 1)]
    for i, (s, dim) in enumerate(specs):
        x = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=1510 + i)
        out.append(SampleInput(x, dim=dim))
    return out


def sample_normalize(op_info, device, dtype, requires_grad):
    # vectors with norm comfortably > 0 (low/high keep them from collapsing) so the
    # quotient gradient is smooth.
    out = []
    specs = [((3, 4), 1), ((2, 5), -1), ((2, 3, 4), 2)]
    for i, (s, dim) in enumerate(specs):
        x = make_tensor(*s, dtype=dtype, low=0.3, high=2.0,
                        requires_grad=requires_grad, seed=1520 + i)
        out.append(SampleInput(x, dim=dim))
    return out


def sample_cosine_similarity(op_info, device, dtype, requires_grad):
    out = []
    specs = [((3, 4), 1), ((2, 5), -1)]
    for i, (s, dim) in enumerate(specs):
        x1 = make_tensor(*s, dtype=dtype, low=0.3, high=2.0,
                         requires_grad=requires_grad, seed=1530 + i)
        x2 = make_tensor(*s, dtype=dtype, low=0.3, high=2.0,
                         requires_grad=requires_grad, seed=1540 + i)
        out.append(SampleInput(x1, x2, dim=dim))
    return out


op_db = [
    # smooth activation -> gradgrad stays on
    UnaryUfuncInfo("log_sigmoid", ref=log_sigmoid_ref, op=nn.log_sigmoid),

    # piecewise clamp -> gradgrad off (2nd derivative differentiates the step mask)
    OpInfo("hardtanh", op=nn.hardtanh, ref=hardtanh_ref,
           sample_inputs_func=sample_hardtanh, supports_gradgrad=False),

    # gated / normalized: smooth on the sampled region
    OpInfo("glu", op=nn.glu, ref=glu_ref, sample_inputs_func=sample_glu),
    OpInfo("normalize", op=nn.normalize, ref=normalize_ref,
           sample_inputs_func=sample_normalize),
    OpInfo("cosine_similarity", op=nn.cosine_similarity, ref=cosine_similarity_ref,
           sample_inputs_func=sample_cosine_similarity),
]
