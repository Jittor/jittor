# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Padding OpInfos: ``F.pad`` modes constant / reflect / replicate / circular.

Padding is a *linear* operator in its input, so the high-value target here is the
BACKWARD: the gradient of a non-constant pad must FOLD the cotangent of every padded
cell back onto the source cell it was copied from. A wrong fold is silent -- the
forward still matches np.pad while the gradient quietly drops (or misplaces) the
boundary contributions. ``reflect``/``circular`` are the dangerous ones: their
boundary cells map to *interior* source cells, so the adjoint scatter-adds several
output cells onto one input cell. This module gradchecks each mode so that fold is
pinned in float64.

Forward is pinned to an INDEPENDENT numpy reference (np.pad with the matching mode),
adapted verbatim from the validated ``test_torch_compat_pad`` references -- including
the torch padding-tuple convention: ``F.pad(x, (l, r, t, b))`` pads the LAST dim by
(l, r) and the 2nd-to-last by (t, b) (reversed, trailing-dim-first). np.pad's mode
names differ: replicate->edge, reflect->reflect, circular->wrap, constant->constant.

Differentiation contract (see ``test_ops._diff_plan``): only ``input`` and floating
positional ``args`` are differentiated. ``padding`` is a python tuple (not a Var) and
``mode``/``value`` are kwargs, so all of them are held fixed -- exactly ``input`` is
the differentiated leaf, which is what we want (pad is not differentiable w.r.t. its
integer pad amounts). All four modes are linear maps of ``input``, hence smooth
everywhere with a constant Jacobian -> both gradcheck and gradgradcheck are
meaningful and supported.

Samples are kept tiny ((1,1,4,4) with pad (1,1,1,1) = 16 differentiated elements) so
the O(numel) float64 finite-difference gradcheck stays cheap; reflect/circular pads
are chosen strictly smaller than the corresponding input dim (np.pad 'reflect'/'wrap'
and jittor's reindex both require pad < dim for a single fold).
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs
# np.pad mode names differ from torch's; map them here (validated mapping, copied
# from test_torch_compat_pad._NP_MODE).
_NP_MODE = {"constant": "constant", "replicate": "edge",
            "reflect": "reflect", "circular": "wrap"}


def pad_ref(x, padding, mode="constant", value=0.0):
    """Independent oracle for ``F.pad`` -- adapted verbatim from
    ``test_torch_compat_pad.pad_ref``.

    ``padding`` is the torch tuple: reversed and applied to the TRAILING dims.
    ``padding=(l,r)`` pads the last dim; ``(l,r,t,b)`` the last two;
    ``(l,r,t,b,f,k)`` the last three. Leading dims get (0, 0).
    """
    pad = list(padding)
    npairs = len(pad) // 2
    widths = [(0, 0)] * (x.ndim - npairs)
    for i in range(npairs):
        # pad[0:2] is the LAST dim, pad[2:4] the previous, ... -> reverse pair order.
        lo = pad[2 * (npairs - 1 - i)]
        hi = pad[2 * (npairs - 1 - i) + 1]
        widths.append((lo, hi))
    if mode == "constant":
        return np.pad(x, widths, mode="constant", constant_values=value)
    return np.pad(x, widths, mode=_NP_MODE[mode])


# --------------------------------------------------------------- sample builders
# Each sample differentiates ONLY `input` (the pad tuple is positional-but-non-Var
# and mode/value are kwargs, so both are held fixed). Inputs are <= 32 elements.

def _make(shape, seed, dtype, requires_grad):
    return make_tensor(*shape, dtype=dtype, low=-2.0, high=2.0,
                       requires_grad=requires_grad, seed=seed)


def sample_constant(op_info, device, dtype, requires_grad):
    """constant pad: linear, every boundary is a fresh `value` cell (no fold).

    NB: the fill values are kept INTEGER-valued on purpose. A *fractional* fill (e.g.
    0.7) triggers a real jittor CPU-codegen bug -- the asm_tuner mangles the
    ``itof(0x...)`` hex-float constant of the reindex overflow value into a malformed
    assembly literal ("exponent has no digits") so the kernel fails to compile on CPU
    (CUDA is fine). That bug is pinned explicitly (and loudly) by the xfail test
    ``test_kernel_traps.TestKernelTraps.test_constant_pad_fractional_fill_cpu_asmtuner``;
    here we keep integer fills so the pad SEMANTICS stay covered on CPU + CUDA."""
    out = []
    # canonical (1,1,4,4) symmetric, non-zero value.
    out.append(SampleInput(_make((1, 1, 4, 4), 900, dtype, requires_grad),
                           (1, 1, 1, 1), mode="constant", value=2.0))
    # last-dim-only asymmetric, default value (0).
    out.append(SampleInput(_make((1, 1, 4, 4), 901, dtype, requires_grad),
                           (2, 1), mode="constant", value=0.0))
    # asymmetric over the last two dims incl. zero-on-one-side.
    out.append(SampleInput(_make((1, 1, 4, 4), 902, dtype, requires_grad),
                           (0, 2, 1, 0), mode="constant", value=-3.0))
    return out


def sample_replicate(op_info, device, dtype, requires_grad):
    """replicate (np 'edge'): boundary cells fold onto the first/last source cell."""
    out = []
    out.append(SampleInput(_make((1, 1, 4, 4), 910, dtype, requires_grad),
                           (1, 1, 1, 1), mode="replicate"))
    out.append(SampleInput(_make((1, 1, 4, 4), 911, dtype, requires_grad),
                           (2, 1), mode="replicate"))                # last dim only
    out.append(SampleInput(_make((1, 1, 4, 4), 912, dtype, requires_grad),
                           (2, 1, 3, 1), mode="replicate"))          # asymmetric (< dim)
    return out


def sample_reflect(op_info, device, dtype, requires_grad):
    """reflect: boundary cells fold onto INTERIOR source cells -- the silent-fold case.

    np.pad 'reflect' (and jittor's reindex) require pad < dim, so every side is < 4.
    """
    out = []
    out.append(SampleInput(_make((1, 1, 4, 4), 920, dtype, requires_grad),
                           (1, 1, 1, 1), mode="reflect"))
    out.append(SampleInput(_make((1, 1, 4, 4), 921, dtype, requires_grad),
                           (2, 1), mode="reflect"))                  # last dim only
    out.append(SampleInput(_make((1, 1, 4, 4), 922, dtype, requires_grad),
                           (2, 1, 3, 2), mode="reflect"))            # asymmetric (< 4)
    return out


def sample_circular(op_info, device, dtype, requires_grad):
    """circular (np 'wrap'): boundary cells fold onto the OPPOSITE-edge source cells.

    jittor's single-fold reindex needs pad < dim; keep every side < 4.
    """
    out = []
    out.append(SampleInput(_make((1, 1, 4, 4), 930, dtype, requires_grad),
                           (1, 1, 1, 1), mode="circular"))
    out.append(SampleInput(_make((1, 1, 4, 4), 931, dtype, requires_grad),
                           (2, 1), mode="circular"))                 # last dim only
    out.append(SampleInput(_make((1, 1, 4, 4), 932, dtype, requires_grad),
                           (2, 1, 1, 2), mode="circular"))           # asymmetric (< 4)
    return out


# --------------------------------------------------------------------- op_db
# pad is LINEAR in `input` -> constant Jacobian -> gradcheck AND gradgradcheck are
# both meaningful and supported for every mode. The four variants share one op
# (F.pad) and one ref (pad_ref), differing only by the `mode` baked into each sample.

op_db = [
    OpInfo("pad", variant_test_name="constant", op=F.pad, ref=pad_ref,
           sample_inputs_func=sample_constant),
    OpInfo("pad", variant_test_name="replicate", op=F.pad, ref=pad_ref,
           sample_inputs_func=sample_replicate),
    OpInfo("pad", variant_test_name="reflect", op=F.pad, ref=pad_ref,
           sample_inputs_func=sample_reflect),
    OpInfo("pad", variant_test_name="circular", op=F.pad, ref=pad_ref,
           sample_inputs_func=sample_circular),
]
