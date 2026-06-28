# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Extra reduction OpInfos: max/min(dim)->namedtuple, amax/amin, var/std, cumsum/
cumprod, logsumexp, norm(p=2), and the non-differentiable argmax/argmin/all/any/
count_nonzero.

These close the reduction *backward* holes the core registry doesn't reach:
  * max/min with a ``dim`` return a (values, indices) namedtuple; the numpy ref
    returns the VALUES and ``test_ops`` compares ``.values`` automatically.
  * var/std are tested with jittor's torch-compat default (UNBIASED, ddof=1) -> the
    numpy refs pass ``ddof=1``.
  * argmax/argmin/all/any/count_nonzero are integer/bool-valued -> not
    differentiable (``supports_autograd=False``); they only run the forward battery.

Signatures are taken from jittor's torch-compat layer (the variants installed on
``import jittor``): see ``torch_compat.py`` (amax/amin/argmax/argmin/count_nonzero/
norm/logsumexp/cumsum/cumprod/var/std/all/any) and ``__init__.py``/``misc.py``/
``nn.py`` for the natives. The torch-compat reductions spell the keep-dim flag
``keepdim`` (no ``s``); each sample func below uses the exact name its op accepts.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs

def _atleast1d(a):
    """jittor has no 0-d scalar: a full reduction yields a (1,)-shaped Var, so the
    reference's python/0-d scalar must be lifted to 1-D to match shapes exactly."""
    return np.atleast_1d(a)


def amax_ref(x, dim=None, keepdim=False):
    return _atleast1d(np.max(x, axis=dim, keepdims=keepdim))


def amin_ref(x, dim=None, keepdim=False):
    return _atleast1d(np.min(x, axis=dim, keepdims=keepdim))


def maxdim_ref(x, dim, keepdim=False):
    """torch max(input, dim) -> (values, indices); test_ops compares .values, so the
    ref returns the VALUES only (np.max along dim)."""
    return np.max(x, axis=dim, keepdims=keepdim)


def mindim_ref(x, dim, keepdim=False):
    return np.min(x, axis=dim, keepdims=keepdim)


def var_ref(x, dim=None, keepdim=False):
    # jittor torch-compat var defaults to UNBIASED (Bessel correction) -> ddof=1.
    return _atleast1d(np.var(x, axis=dim, ddof=1, keepdims=keepdim))


def std_ref(x, dim=None, keepdim=False):
    return _atleast1d(np.std(x, axis=dim, ddof=1, keepdims=keepdim))


def cumsum_ref(x, dim=-1):
    return np.cumsum(x, axis=dim)


def cumprod_ref(x, dim=-1):
    return np.cumprod(x, axis=dim)


def logsumexp_ref(x, dim, keepdim=False):
    m = np.max(x, axis=dim, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=dim, keepdims=True))
    if keepdim:
        return out
    return np.squeeze(out, axis=dim)


def norm2_ref(x, p=2, dim=None, keepdim=False):
    # torch/torch-compat norm: p=2 (Euclidean); dim=None reduces over the flattened
    # tensor to a (1,)-shaped scalar, an int dim reduces that axis.
    if dim is None:
        return _atleast1d(np.sqrt(np.sum(np.square(x.reshape(-1)))))
    return np.sqrt(np.sum(np.square(x), axis=dim, keepdims=keepdim))


def argmax_ref(x, dim=None, keepdim=False):
    out = np.argmax(x, axis=dim)
    if keepdim and dim is not None:
        out = np.expand_dims(out, axis=dim)
    return _atleast1d(out)


def argmin_ref(x, dim=None, keepdim=False):
    out = np.argmin(x, axis=dim)
    if keepdim and dim is not None:
        out = np.expand_dims(out, axis=dim)
    return _atleast1d(out)


def all_ref(x, dim=None, keepdim=False):
    return _atleast1d(np.all(x != 0, axis=dim, keepdims=keepdim))


def any_ref(x, dim=None, keepdim=False):
    return _atleast1d(np.any(x != 0, axis=dim, keepdims=keepdim))


def count_nonzero_ref(x, dim=None):
    return _atleast1d(np.count_nonzero(x, axis=dim))


# --------------------------------------------------------------- sample builders
# Small tensors only: gradcheck is O(numel) forward passes. The differentiated
# operand stays <= 24 elements (a 2x3x4 = 24 tensor or smaller).

_RED_SHAPE = (2, 3, 4)   # 24 elements


def _keepdim_sweep(op_info, device, dtype, requires_grad, seed0, dims=(0, 1, 2, -1)):
    """Full reduce + a (dim, keepdim) sweep, spelling the flag ``keepdim`` (the
    torch-compat reductions' kwarg name)."""
    out = [SampleInput(make_tensor(*_RED_SHAPE, dtype=dtype,
                                   requires_grad=requires_grad, seed=seed0))]
    for i, dim in enumerate(dims):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=dtype, requires_grad=requires_grad,
                            seed=seed0 + 1 + i),
                dim=dim, keepdim=keepdim))
    return out


def sample_amax(op_info, device, dtype, requires_grad):
    return _keepdim_sweep(op_info, device, dtype, requires_grad, seed0=600)


def sample_amin(op_info, device, dtype, requires_grad):
    return _keepdim_sweep(op_info, device, dtype, requires_grad, seed0=610)


def sample_var(op_info, device, dtype, requires_grad):
    return _keepdim_sweep(op_info, device, dtype, requires_grad, seed0=620)


def sample_std(op_info, device, dtype, requires_grad):
    return _keepdim_sweep(op_info, device, dtype, requires_grad, seed0=630)


def sample_maxmin_dim(op_info, device, dtype, requires_grad, seed0):
    """max/min with a REQUIRED dim -> namedtuple. No full-reduce sample (that path
    returns a bare values-only Var, not the namedtuple). Sweeps dim/keepdim."""
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=dtype, requires_grad=requires_grad,
                            seed=seed0 + i),
                dim=dim, keepdim=keepdim))
    return out


def sample_max_dim(op_info, device, dtype, requires_grad):
    return sample_maxmin_dim(op_info, device, dtype, requires_grad, seed0=640)


def sample_min_dim(op_info, device, dtype, requires_grad):
    return sample_maxmin_dim(op_info, device, dtype, requires_grad, seed0=650)


def sample_cumsum(op_info, device, dtype, requires_grad):
    # cumsum/cumprod take a positional `dim` (default -1); shape is preserved.
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        out.append(SampleInput(
            make_tensor(*_RED_SHAPE, dtype=dtype, requires_grad=requires_grad,
                        seed=660 + i),
            dim=dim))
    return out


def sample_cumprod(op_info, device, dtype, requires_grad):
    # Keep entries strictly POSITIVE: jittor's sign-aware cumprod (misc.py) routes
    # through cumsum(log|x|) with a sign-parity term whose derivative wrt sign is 0;
    # a clean positive domain makes the cumprod analytic gradient match numpy.
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        out.append(SampleInput(
            make_tensor(*_RED_SHAPE, dtype=dtype, low=0.3, high=2.0,
                        requires_grad=requires_grad, seed=670 + i),
            dim=dim))
    return out


def sample_logsumexp(op_info, device, dtype, requires_grad):
    # logsumexp(input, dim, keepdim=False): dim is REQUIRED (no full-reduce form).
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=dtype, requires_grad=requires_grad,
                            seed=680 + i),
                dim=dim, keepdim=keepdim))
    return out


def sample_norm2(op_info, device, dtype, requires_grad):
    # norm(input, p=2, dim=None, keepdim=False). Pass p as a kwarg (non-tensor) so it
    # isn't differentiated. low>0 keeps the squared-sum strictly positive so the
    # sqrt() backward is finite (no division by a zero norm).
    out = [SampleInput(make_tensor(*_RED_SHAPE, dtype=dtype, low=0.2, high=2.0,
                                   requires_grad=requires_grad, seed=690), p=2)]
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=dtype, low=0.2, high=2.0,
                            requires_grad=requires_grad, seed=691 + i),
                p=2, dim=dim, keepdim=keepdim))
    return out


def sample_argmaxmin(op_info, device, dtype, requires_grad, seed0):
    # Non-differentiable: requires_grad is ignored by the forward-only battery. Use a
    # REQUIRED dim (avoid the dim=None flatten path, whose scalar/tie shape is
    # ambiguous) and float inputs so ties are vanishingly unlikely -> jittor's
    # first-max index matches numpy's.
    out = []
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype=dtype, seed=seed0 + i),
                dim=dim, keepdim=keepdim))
    return out


def sample_argmax(op_info, device, dtype, requires_grad):
    return sample_argmaxmin(op_info, device, dtype, requires_grad, seed0=700)


def sample_argmin(op_info, device, dtype, requires_grad):
    return sample_argmaxmin(op_info, device, dtype, requires_grad, seed0=710)


def sample_allany(op_info, device, dtype, requires_grad, seed0):
    # all/any reduce truthiness. Build 0/1 integer tensors so the bool result has a
    # mix of True/False (a float tensor in [-9,9] is nonzero almost everywhere ->
    # trivially all-True). dtype here is the test's chosen float dtype; override with
    # int32 0/1 values via low/high so the (x != 0) ref is unambiguous.
    out = [SampleInput(make_tensor(*_RED_SHAPE, dtype="int32", low=0, high=2,
                                   seed=seed0))]
    for i, dim in enumerate([0, 1, 2, -1]):
        for keepdim in (False, True):
            out.append(SampleInput(
                make_tensor(*_RED_SHAPE, dtype="int32", low=0, high=2,
                            seed=seed0 + 1 + i),
                dim=dim, keepdim=keepdim))
    return out


def sample_all(op_info, device, dtype, requires_grad):
    return sample_allany(op_info, device, dtype, requires_grad, seed0=720)


def sample_any(op_info, device, dtype, requires_grad):
    return sample_allany(op_info, device, dtype, requires_grad, seed0=730)


def sample_count_nonzero(op_info, device, dtype, requires_grad):
    # count_nonzero(x, dim=None): no keepdim param in the torch-compat impl. Use 0/1
    # int tensors so the count is meaningful (not trivially numel).
    out = [SampleInput(make_tensor(*_RED_SHAPE, dtype="int32", low=0, high=2,
                                   seed=740))]
    for i, dim in enumerate([0, 1, 2, -1]):
        out.append(SampleInput(
            make_tensor(*_RED_SHAPE, dtype="int32", low=0, high=2, seed=741 + i),
            dim=dim))
    return out


op_db = [
    # ---- max / min with dim -> (values, indices) namedtuple --------------------
    # supports_gradgrad=False: the backward scatters the upstream grad onto the
    # argmax positions (a piecewise-constant selection); its 2nd derivative is 0/
    # ill-defined, so gradgradcheck is not meaningful (torch declares the same).
    OpInfo("max", op=jt.max, ref=maxdim_ref, sample_inputs_func=sample_max_dim,
           variant_test_name="reduction_with_dim", supports_gradgrad=False),
    OpInfo("min", op=jt.min, ref=mindim_ref, sample_inputs_func=sample_min_dim,
           variant_test_name="reduction_with_dim", supports_gradgrad=False),

    # ---- amax / amin (values-only reductions; same select-backward) ------------
    OpInfo("amax", op=jt.amax, ref=amax_ref, sample_inputs_func=sample_amax,
           supports_gradgrad=False),
    OpInfo("amin", op=jt.amin, ref=amin_ref, sample_inputs_func=sample_amin,
           supports_gradgrad=False),

    # ---- variance / std (UNBIASED by default in jittor's torch-compat) ---------
    OpInfo("var", op=jt.var, ref=var_ref, sample_inputs_func=sample_var),
    OpInfo("std", op=jt.std, ref=std_ref, sample_inputs_func=sample_std),

    # ---- cumulative reductions -------------------------------------------------
    # cumsum: numpy_code op with an explicit (flip-cumsum) backward.
    # supports_gradgrad=False: FINDING -- 2nd-order autodiff through a numpy_code op
    # SEGFAULTS in jittor's C++ ``NumpyCodeOp::grad`` (the backward is itself a numpy_code
    # op carrying no registered grad, so grad-of-grad dereferences null). A crash, not a
    # silent-wrong; 1st-order gradcheck passes. Same class as cumprod below.
    OpInfo("cumsum", op=jt.cumsum, ref=cumsum_ref, sample_inputs_func=sample_cumsum,
           supports_gradgrad=False),
    # cumprod: sign-aware exp(cumsum(log|x|)); positive-only samples (see sampler).
    # supports_gradgrad=False: the magnitude path's 2nd derivative through the
    # cumsum-of-logs composition is not reliably differentiable in jittor.
    OpInfo("cumprod", op=jt.cumprod, ref=cumprod_ref,
           sample_inputs_func=sample_cumprod, supports_gradgrad=False),

    # ---- logsumexp -------------------------------------------------------------
    OpInfo("logsumexp", op=jt.logsumexp, ref=logsumexp_ref,
           sample_inputs_func=sample_logsumexp),

    # ---- norm (p=2 / Euclidean) ------------------------------------------------
    # supports_gradgrad=False: norm's backward divides by the (data-dependent) norm
    # value; its 2nd derivative is not reliably available (mirrors core's norm note).
    OpInfo("norm", op=jt.norm, ref=norm2_ref, sample_inputs_func=sample_norm2,
           variant_test_name="p2", supports_gradgrad=False),

    # ---- non-differentiable (integer / bool valued) ----------------------------
    OpInfo("argmax", op=jt.argmax, ref=argmax_ref, sample_inputs_func=sample_argmax,
           dtypes=cu.floating_types(), supports_autograd=False),
    OpInfo("argmin", op=jt.argmin, ref=argmin_ref, sample_inputs_func=sample_argmin,
           dtypes=cu.floating_types(), supports_autograd=False),
    OpInfo("all", op=jt.all, ref=all_ref, sample_inputs_func=sample_all,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("any", op=jt.any, ref=any_ref, sample_inputs_func=sample_any,
           dtypes=cu.integral_types(), supports_autograd=False),
    OpInfo("count_nonzero", op=jt.count_nonzero, ref=count_nonzero_ref,
           sample_inputs_func=sample_count_nonzero,
           dtypes=cu.integral_types(), supports_autograd=False),
]
