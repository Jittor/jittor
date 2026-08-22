# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Structural / linear-algebra-shaped OpInfos that live in jittor core
(``misc.py`` / ``nn.py``) but had no op_db entry:

  * ``tril`` / ``triu``  -- triangular masks (a mask-multiply; linear, gradchecked),
  * ``cross``            -- 3-vector cross product (bilinear; both operands gradchecked),
  * ``kron``             -- Kronecker product (bilinear; vs ``np.kron``),
  * ``tensordot``        -- general tensor contraction (vs ``np.tensordot``),
  * ``cummax`` / ``cummin`` -- prefix max/min SCAN, returning a (values, indices)
    namedtuple. Tested forward-only against ``np.maximum/minimum.accumulate``: the scan
    is the interesting kernel, and -- crucially -- ``test_device_parity`` then compares
    the CUDA scan against the CPU scan (a prefix-scan is a classic accelerator
    silent-wrong spot). The ``.values`` field is what both drivers compare.

Each op has an INDEPENDENT numpy reference. Differentiable ops keep samples small
(gradcheck is O(numel) forward passes) and on a smooth region.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo


# ----------------------------------------------------------------- numpy refs
def tril_ref(x, diagonal=0):
    return np.tril(x, diagonal)


def triu_ref(x, diagonal=0):
    return np.triu(x, diagonal)


def cross_ref(x, y, dim=-1):
    return np.cross(x, y, axis=dim)


def kron_ref(x, y):
    return np.kron(x, y)


def tensordot_ref(x, y, dims=2):
    return np.tensordot(x, y, axes=dims)


def cummax_ref(x, dim=0):
    return np.maximum.accumulate(x, axis=dim)


def cummin_ref(x, dim=0):
    return np.minimum.accumulate(x, axis=dim)


def meshgrid_ref(x, y):
    # 'ij' indexing (jittor's meshgrid convention). Stack the grids into one array so the
    # multi-output is comparable as a single tensor (the grid KERNEL is what we pin).
    return np.stack(np.meshgrid(x, y, indexing="ij"))


# ----------------------------------------------------------------- jittor callables
def _cummax(x, dim=0):
    return jt.cummax(x, dim=dim)


def _cummin(x, dim=0):
    return jt.cummin(x, dim=dim)


# --------------------------------------------------------------- sample builders
def sample_triangular(op_info, device, dtype, requires_grad):
    # square + non-square + batched, swept over a few diagonals (incl. +/- offsets).
    out = []
    shapes = [(4, 4), (3, 5), (2, 3, 4)]
    diags = [0, 1, -1]
    for i, s in enumerate(shapes):
        for j, d in enumerate(diags):
            out.append(SampleInput(
                make_tensor(*s, dtype=dtype, requires_grad=requires_grad,
                            seed=1400 + 10 * i + j),
                diagonal=d))
    return out


def sample_cross(op_info, device, dtype, requires_grad):
    # cross product needs the contracted axis to be length 3; differentiate both.
    # NB: skip a bare 1-D (3,) input -- numpy returns shape (3,) there while jittor keeps
    # a trailing (3,1); the VALUES agree but the shape convention differs, so we test only
    # multi-axis shapes (which match exactly) and let that convention gap be out of scope.
    out = []
    shapes = [(2, 3), (5, 3), (4, 3, 2)]
    dims = [-1, -1, 1]
    for i, (s, dim) in enumerate(zip(shapes, dims)):
        a = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=1430 + i)
        b = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=1440 + i)
        out.append(SampleInput(a, b, dim=dim))
    return out


def sample_kron(op_info, device, dtype, requires_grad):
    out = []
    pairs = [((2, 2), (2, 2)), ((2, 3), (3, 1)), ((3,), (2,))]
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, requires_grad=requires_grad, seed=1450 + i)
        b = make_tensor(*sb, dtype=dtype, requires_grad=requires_grad, seed=1460 + i)
        out.append(SampleInput(a, b))
    return out


def sample_tensordot(op_info, device, dtype, requires_grad):
    out = []
    # (shapeA, shapeB, dims) chosen so the contraction is conformable.
    specs = [((2, 3), (3, 4), 1), ((2, 3, 4), (4, 5), 1), ((2, 3), (2, 3), 2)]
    for i, (sa, sb, dims) in enumerate(specs):
        a = make_tensor(*sa, dtype=dtype, requires_grad=requires_grad, seed=1470 + i)
        b = make_tensor(*sb, dtype=dtype, requires_grad=requires_grad, seed=1480 + i)
        out.append(SampleInput(a, b, dims=dims))
    return out


def sample_cumscan(op_info, device, dtype, requires_grad):
    # forward-only prefix scan; sweep the scan dim (incl. negative) over a 3-D tensor.
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        out.append(SampleInput(
            make_tensor(3, 4, 5, dtype=dtype, requires_grad=requires_grad, seed=1490 + i),
            dim=dim))
    return out


def _meshgrid(x, y):
    return jt.stack(jt.meshgrid(x, y))


def sample_meshgrid(op_info, device, dtype, requires_grad):
    out = []
    pairs = [((3,), (2,)), ((4,), (5,)), ((2,), (3,))]
    for i, (sa, sb) in enumerate(pairs):
        a = make_tensor(*sa, dtype=dtype, requires_grad=requires_grad, seed=1495 + i)
        b = make_tensor(*sb, dtype=dtype, requires_grad=requires_grad, seed=1497 + i)
        out.append(SampleInput(a, b))
    return out


op_db = [
    # ---- triangular masks: linear in the input, gradchecked ----
    OpInfo("tril", op=jt.tril, ref=tril_ref, sample_inputs_func=sample_triangular),
    OpInfo("triu", op=jt.triu, ref=triu_ref, sample_inputs_func=sample_triangular),

    # ---- bilinear contractions: both operands differentiated ----
    OpInfo("cross", op=jt.cross, ref=cross_ref, sample_inputs_func=sample_cross),
    OpInfo("kron", op=jt.kron, ref=kron_ref, sample_inputs_func=sample_kron),
    OpInfo("tensordot", op=jt.tensordot, ref=tensordot_ref,
           sample_inputs_func=sample_tensordot),

    # ---- prefix-scan (values, indices) namedtuple -- forward + device parity only ----
    OpInfo("cummax", op=_cummax, ref=cummax_ref, sample_inputs_func=sample_cumscan,
           supports_autograd=False),
    OpInfo("cummin", op=_cummin, ref=cummin_ref, sample_inputs_func=sample_cumscan,
           supports_autograd=False),

    # ---- meshgrid ('ij'): multi-grid broadcast, stacked -- forward + parity ----
    OpInfo("meshgrid", op=_meshgrid, ref=meshgrid_ref, sample_inputs_func=sample_meshgrid,
           supports_autograd=False),
]
