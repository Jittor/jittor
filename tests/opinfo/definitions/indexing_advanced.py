# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Differentiable ``getitem`` (advanced indexing) OpInfos.

The companion to ``indexing.py``: where that module covers the *named* gather/scatter
family, this one exercises the **indexing kernel itself** -- ``x[idx]`` -- and, just as
importantly, its backward, which is a *scatter-add* of the upstream cotangent into a
zeroed buffer at the very same positions (``getitem_op``'s grad is a ``setitem``). A
green ``test_gradcheck`` here therefore proves the scatter-back kernel routes every
cotangent element to the correct source coordinate, for each index *kind* below; the
generic device-parity driver then re-runs the same forward+backward on CPU vs CUDA/NPU
so a kernel that is right on one device and wrong on another is caught.

Index kinds covered (one OpInfo each, ``variant_test_name`` disambiguated):

  * ``getitem_slice``    -- basic slice ``x[1:3]`` (contiguous view).
  * ``getitem_int``      -- integer index ``x[2]`` (drops a dim; on >=2-D so the result
                            is non-scalar and matches numpy exactly).
  * ``getitem_step``     -- strided slice ``x[::2]`` (non-unit step -> strided scatter
                            back; a dropped-element backward would read as a mismatch).
  * ``getitem_mask``     -- BOOLEAN MASK ``x[mask]`` -> 1-D gather; grad scatters the
                            cotangent back into the masked positions. Mask is a FIXED
                            bool Var (independent of x) so the selected shape is stable
                            under gradcheck's finite-difference perturbations.
  * ``getitem_fancy_1d`` -- fancy/advanced index ``x[[0,2,1]]`` (reorder rows; a repeat
                            ``[0,0,2]`` lands the accumulate path of the scatter-back).
  * ``getitem_fancy_2d`` -- paired advanced index ``x[rows, cols]`` (point gather; grad
                            is np.add.at into the (row,col) cells, duplicates accumulate).
  * ``getitem_ellipsis`` -- ``x[..., 0]`` / ``x[..., None, :]`` (ellipsis + newaxis: the
                            kernel's axis bookkeeping, newaxis adds a size-1 out dim).
  * ``getitem_negative``  -- NEGATIVE indices ``x[-1]`` and ``x[:, -2:]``. Negative
                            *advanced* (var/list) indices had a backward bug -- the
                            setitem (grad) kernel did not normalize ``iid<0`` the way
                            getitem does, so the cotangent scattered out of bounds and
                            the indexed rows got no gradient (fixed 58e95b73). The
                            ``getitem_fancy_neg`` variant locks the var/list negative
                            path specifically (``x[[-1, -2]]``), which is the one that
                            actually reached the kernel; ``x[-1]`` / ``x[:, -2:]`` are
                            int/slice negatives (host-normalized in infer_slices).

The index is BAKED INTO the op lambda (``op=lambda x: x[1:3]``), NOT passed as a
SampleInput arg -- so the only operand the gradcheck driver differentiates is the float
``input`` (see ``test_ops._diff_plan``: it differentiates ``input`` plus any float Var
in ``args``; there are no args here). The numpy ``ref`` mirrors the identical indexing
on the numpy array, giving an INDEPENDENT forward oracle. Differentiated tensors are
kept small (<= 24 elems) because gradcheck cost is O(numel).

Shape note: jittor has no 0-d scalar, so an integer index that would yield a numpy
``()`` scalar yields jittor ``(1,)`` instead. Every integer/negative-int sample here
indexes a >=2-D Var (or a column-slice) so the result is genuinely non-scalar and the
numpy reference matches jittor's shape exactly -- no ``atleast_1d`` papering-over, the
oracle stays honest.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# --------------------------------------------------------------------------- ops
#
# Each op bakes a fixed index into a lambda so the SampleInput carries only the float
# `input` Var. For the fancy/mask kinds the addressing operand is captured as a
# module-level int64/bool array (NOT closed over a per-call Var) so the SAME constant
# index is used by both the jittor op and the numpy ref, and is identical across all of
# gradcheck's perturbed forward passes (a per-call random index would make the
# finite-difference forward inconsistent and the Jacobian meaningless).

# fancy-index row orders (include a repeat to exercise the accumulating scatter-back).
_ROWS_REORDER = [0, 2, 1]            # permutation (no collision)
_ROWS_REPEAT  = [0, 0, 2]            # repeat row 0 -> grad accumulates into row 0
_ROWS_NEG     = [-1, -2]             # negative var/list index (the 58e95b73 path)

# paired advanced index x[rows, cols] -> picks points (rows[k], cols[k]); the repeated
# (1,1) cell forces the duplicate-destination accumulate in the backward scatter.
_PAIR_ROWS = jt.array(np.array([0, 1, 2, 1], dtype="int64"), dtype="int64")
_PAIR_COLS = jt.array(np.array([3, 0, 2, 0], dtype="int64"), dtype="int64")
_PAIR_ROWS_NP = np.array([0, 1, 2, 1])
_PAIR_COLS_NP = np.array([3, 0, 2, 0])


def _fixed_bool_mask(*shape, seed):
    """A deterministic bool mask, independent of the indexed tensor, so the number of
    selected elements (hence the output shape) is constant under gradcheck."""
    rng = np.random.RandomState(seed)
    a = np.ascontiguousarray(rng.uniform(size=shape) > 0.4)
    # guarantee at least one True and one False so the gather is non-degenerate.
    a.flat[0] = True
    a.flat[-1] = False
    return a


# A single mask instance reused by op + ref + every gradcheck forward.
_MASK_3x4 = _fixed_bool_mask(3, 4, seed=701)
_MASK_3x4_JT = jt.array(_MASK_3x4, dtype="bool")


# ------------------------------------------------------------------- numpy refs
# Every ref reproduces the SAME indexing on the numpy array. Kept as named functions
# (not lambdas) so a forward failure names the index kind. The op lambdas must mirror
# these byte-for-byte.

def ref_slice(x):        return x[1:3]
def op_slice(x):         return x[1:3]

def ref_int(x):          return x[2]            # >=2-D input -> non-scalar, matches jittor
def op_int(x):           return x[2]

def ref_step(x):         return x[::2]
def op_step(x):          return x[::2]

def ref_mask(x):         return x[_MASK_3x4]    # C-order 1-D gather, like jittor
def op_mask(x):          return x[_MASK_3x4_JT]

def ref_fancy_1d(x):     return x[_ROWS_REPEAT]
def op_fancy_1d(x):      return x[_ROWS_REPEAT]

def ref_fancy_2d(x):     return x[_PAIR_ROWS_NP, _PAIR_COLS_NP]
def op_fancy_2d(x):      return x[_PAIR_ROWS, _PAIR_COLS]

def ref_ellipsis(x):     return x[..., 0]       # last-axis int index via ellipsis
def op_ellipsis(x):      return x[..., 0]

def ref_newaxis(x):      return x[..., None, :]  # ellipsis + newaxis (adds a size-1 dim)
def op_newaxis(x):       return x[..., None, :]

def ref_neg_int(x):      return x[-1]           # negative int (host-normalized)
def op_neg_int(x):       return x[-1]

def ref_neg_slice(x):    return x[:, -2:]       # negative slice bound
def op_neg_slice(x):     return x[:, -2:]

def ref_fancy_neg(x):    return x[_ROWS_NEG]    # NEGATIVE var/list index (58e95b73 path)
def op_fancy_neg(x):     return x[_ROWS_NEG]


# --------------------------------------------------------------- sample builders
# Each builder differentiates ONLY `input` (a float Var). Shapes are small; the row
# count is large enough that the baked-in indices are in-bounds.

def _x(*shape, dtype, requires_grad, seed):
    return make_tensor(*shape, dtype=dtype, requires_grad=requires_grad, seed=seed)


def sample_slice(op_info, device, dtype, requires_grad):
    # x[1:3] over the first axis of a (4, 3) Var -> (2, 3); a few shapes for coverage.
    return [
        SampleInput(_x(4, 3, dtype=dtype, requires_grad=requires_grad, seed=710)),
        SampleInput(_x(5, dtype=dtype, requires_grad=requires_grad, seed=711)),  # 1-D, x[1:3]->(2,)
    ]


def sample_int(op_info, device, dtype, requires_grad):
    # x[2] on >=2-D so the result is a real (non-scalar) slice that matches numpy.
    return [
        SampleInput(_x(4, 3, dtype=dtype, requires_grad=requires_grad, seed=720)),     # ->(3,)
        SampleInput(_x(4, 2, 3, dtype=dtype, requires_grad=requires_grad, seed=721)),  # ->(2,3)
    ]


def sample_step(op_info, device, dtype, requires_grad):
    # x[::2] -> strided view; (5,*) so the kept indices are {0,2,4}.
    return [
        SampleInput(_x(5, 4, dtype=dtype, requires_grad=requires_grad, seed=730)),  # ->(3,4)
        SampleInput(_x(6, dtype=dtype, requires_grad=requires_grad, seed=731)),      # ->(3,)
    ]


def sample_mask(op_info, device, dtype, requires_grad):
    # x[mask] with a FIXED (3,4) bool mask -> 1-D of the True count.
    return [
        SampleInput(_x(3, 4, dtype=dtype, requires_grad=requires_grad, seed=740)),
    ]


def sample_fancy_1d(op_info, device, dtype, requires_grad):
    # x[[0,0,2]] over the first axis of a (3, 4) Var -> (3, 4); row 0 repeated so the
    # backward accumulates two cotangent rows into source row 0.
    return [
        SampleInput(_x(3, 4, dtype=dtype, requires_grad=requires_grad, seed=750)),
    ]


def sample_fancy_2d(op_info, device, dtype, requires_grad):
    # x[rows, cols] point-gather on a (4, 4) Var -> (4,); the duplicated (1,0) target
    # exercises the duplicate-destination accumulate in the scatter-back.
    return [
        SampleInput(_x(4, 4, dtype=dtype, requires_grad=requires_grad, seed=760)),
    ]


def sample_ellipsis(op_info, device, dtype, requires_grad):
    # x[..., 0] drops the last axis; x[..., None, :] inserts a size-1 dim.
    return [
        SampleInput(_x(2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=770)),
    ]


def sample_newaxis(op_info, device, dtype, requires_grad):
    return [
        SampleInput(_x(2, 4, dtype=dtype, requires_grad=requires_grad, seed=775)),
    ]


def sample_neg_int(op_info, device, dtype, requires_grad):
    # x[-1] on >=2-D -> last row (non-scalar); negative int normalized on the host.
    return [
        SampleInput(_x(4, 3, dtype=dtype, requires_grad=requires_grad, seed=780)),  # ->(3,)
        SampleInput(_x(3, 2, 3, dtype=dtype, requires_grad=requires_grad, seed=781)),  # ->(2,3)
    ]


def sample_neg_slice(op_info, device, dtype, requires_grad):
    # x[:, -2:] -> last two columns; negative slice bound.
    return [
        SampleInput(_x(3, 5, dtype=dtype, requires_grad=requires_grad, seed=785)),  # ->(3,2)
    ]


def sample_fancy_neg(op_info, device, dtype, requires_grad):
    # x[[-1, -2]] -- the negative VAR/LIST advanced index that reached the (previously
    # un-normalizing) setitem grad kernel; regression-locks 58e95b73. (4,3)->(2,3).
    return [
        SampleInput(_x(4, 3, dtype=dtype, requires_grad=requires_grad, seed=790)),
    ]


# ----------------------------------------------------------------------- op_db
# All differentiable (getitem forward is linear in x -> backward is a scatter, and the
# second derivative is the trivial zero, so gradgrad is left ON to lock that too).
# variant_test_name keeps the shared name "getitem" unique per index kind.

op_db = [
    OpInfo("getitem", variant_test_name="slice",
           op=op_slice, ref=ref_slice, sample_inputs_func=sample_slice),
    OpInfo("getitem", variant_test_name="int",
           op=op_int, ref=ref_int, sample_inputs_func=sample_int),
    OpInfo("getitem", variant_test_name="step",
           op=op_step, ref=ref_step, sample_inputs_func=sample_step),

    # boolean mask: data-dependent output shape, but the mask is FIXED so the shape is
    # stable under gradcheck perturbations and the scatter-back is exact.
    OpInfo("getitem", variant_test_name="mask",
           op=op_mask, ref=ref_mask, sample_inputs_func=sample_mask),

    # fancy / advanced indices (repeat + paired) -> accumulating scatter-back.
    OpInfo("getitem", variant_test_name="fancy_1d",
           op=op_fancy_1d, ref=ref_fancy_1d, sample_inputs_func=sample_fancy_1d),
    OpInfo("getitem", variant_test_name="fancy_2d",
           op=op_fancy_2d, ref=ref_fancy_2d, sample_inputs_func=sample_fancy_2d),

    # ellipsis + newaxis.
    OpInfo("getitem", variant_test_name="ellipsis",
           op=op_ellipsis, ref=ref_ellipsis, sample_inputs_func=sample_ellipsis),
    OpInfo("getitem", variant_test_name="newaxis",
           op=op_newaxis, ref=ref_newaxis, sample_inputs_func=sample_newaxis),

    # negative-index cases (the backward-bug-prone family, 58e95b73).
    OpInfo("getitem", variant_test_name="negative_int",
           op=op_neg_int, ref=ref_neg_int, sample_inputs_func=sample_neg_int),
    OpInfo("getitem", variant_test_name="negative_slice",
           op=op_neg_slice, ref=ref_neg_slice, sample_inputs_func=sample_neg_slice),
    OpInfo("getitem", variant_test_name="fancy_negative",
           op=op_fancy_neg, ref=ref_fancy_neg, sample_inputs_func=sample_fancy_neg),
]
