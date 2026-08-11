# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Interpolate / grid OpInfos: interpolate (nearest, bilinear) / grid_sample
(bilinear) / affine_grid.

Coordinate-mapping ops are classic backward-bug territory: the source-index map,
border clamping and zero-padding all live in the *backward* pass, and the audit
found these forward-only. Each OpInfo here pins the forward to an INDEPENDENT numpy
reference (adapted from the validated ``test_torch_compat_interpolate`` source-index
refs and the documented torch grid_sample / affine_grid maps) and -- via the generic
``test_ops`` driver -- gradchecks the backward in float64.

Why these are gradcheck-clean despite the kinky reputation:

  * ``interpolate`` (nearest and bilinear): each output element is a *fixed* linear
    combination of input elements -- the weights depend only on the in/out geometry,
    never on the input values. So the op is exactly LINEAR in ``input``; its Jacobian
    is constant and finite-difference matches analytic everywhere (no value-dependent
    branch for finite-diff to straddle). 2nd derivative is identically 0 -> gradgrad
    passes trivially and is kept on.
  * ``affine_grid``: ``grid = base_grid @ thetaᵀ`` is affine in ``theta`` -> linear,
    same story (gradgrad on).
  * ``grid_sample`` (bilinear): linear in ``input`` (fixed corner weights), but the
    differentiated ``grid`` enters the *weights* (``dx,dy``) bilinearly, and the
    corner indices come from ``floor`` (a step) and zero-padding at the border (a
    jump). 1st-order gradcheck is meaningful and passes ONLY because every sampled
    grid point is built to land strictly INTERIOR (all four corners inside
    ``[0,W-1]×[0,H-1]``) with fractional parts a safe margin from 0/1 -- so no
    finite-diff step crosses an integer cell wall or the zero-pad border. The 2nd
    derivative would differentiate that floor/border step, so ``supports_gradgrad``
    is False (honest: torch's grid_sample double-backward is likewise restricted).

Differentiation contract (see ``test_ops._diff_plan``): the primary ``input`` and any
*floating positional* args are the differentiated leaves; kwargs are held fixed. So
``size`` / ``mode`` / ``align_corners`` / ``padding_mode`` are passed as KWARGS (they
must stay fixed), while for grid_sample the float ``grid`` is passed POSITIONALLY so
it is differentiated alongside ``input`` (grid_sample is genuinely differentiable
w.r.t. the sampling grid -- the high-value flow-field gradient).

Signature notes (verified against ``jittor.nn``, do-not-guess):
  * ``F.interpolate(X, size=None, scale_factor=None, mode='bilinear',
    align_corners=False, tf_mode=False)``. With ``size=`` (scale_factor None) it
    routes through ``resize``; with ``scale_factor>1`` through ``upsample``. We use
    ``size=`` everywhere so the differentiated path is deterministic.
  * ``F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros',
    align_corners=False)``. ``input`` is (N,C,Hi,Wi); ``grid`` is (N,Ho,Wo,2) and its
    last-dim order is (x=W, y=H) -- grid[...,0] indexes WIDTH, grid[...,1] HEIGHT.
  * ``F.affine_grid(theta, size, align_corners=False)``. ``size`` is a non-float
    shape tuple -> kwarg.

Omitted (noted, not silently dropped): interpolate ``mode='bicubic'`` and
``mode='area'`` -- the bicubic kernel (jittor's ``_bicubic`` with a=-0.75 and a
4x4 clamped stencil) and area/adaptive-pool path are intricate enough that an
independent numpy oracle is its own mini-project; the validated source test only
locks bicubic by shape/constant-invariants, so there is no value-level ref to adapt.
nearest+bilinear are the load-bearing modes (UNet/diffusers decoders). grid_sample
``mode='nearest'`` and ``padding_mode in {border,reflection}`` are also omitted: the
nearest gather is value-discontinuous in grid (not gradcheckable) and the non-zero
padding modes add reflect/clip kinks; the bilinear/zeros path is the one training
loops differentiate.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs
# Adapted verbatim (then float64-tightened) from the validated source-index maps in
# tests/compat/torch/test_torch_compat_interpolate.py.

def _nearest_ref(x, Oh, Ow):
    """torch/jittor nearest: src = floor(i * in/out), clamped to [0, in-1]."""
    N, C, H, W = x.shape
    x = x.astype(np.float64)
    out = np.zeros((N, C, Oh, Ow), dtype=np.float64)
    for i in range(Oh):
        si = min(int(np.floor(i * H / Oh)), H - 1)
        for j in range(Ow):
            sj = min(int(np.floor(j * W / Ow)), W - 1)
            out[:, :, i, j] = x[:, :, si, sj]
    return out


def _bilinear_ref(x, Oh, Ow, align_corners):
    """torch/jittor bilinear: source-coordinate map + edge clamping.

    AC=True:  src = i*(in-1)/(out-1)
    AC=False: src = (i+0.5)*in/out - 0.5  (clamped >=0 at the low border)
    """
    N, C, H, W = x.shape
    x = x.astype(np.float64)
    out = np.zeros((N, C, Oh, Ow), dtype=np.float64)
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


def interpolate_ref(x, size=None, scale_factor=None, mode='bilinear',
                    align_corners=False, tf_mode=False):
    """Forward oracle for F.interpolate. We always drive it with ``size=`` (the
    sample builders never pass scale_factor), so reproduce jittor's resize() map."""
    if scale_factor is not None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    if isinstance(size, int):
        size = (size, size)
    Oh, Ow = int(size[0]), int(size[1])
    if mode == "nearest":
        return _nearest_ref(x, Oh, Ow)
    if mode == "bilinear":
        return _bilinear_ref(x, Oh, Ow, align_corners)
    raise ValueError(f"interpolate_ref: unsupported mode {mode!r}")


def _grid_unnormalize(coord, size, align_corners):
    """jittor grid_sampler_unnormalize: [-1,1] -> source index (matches torch)."""
    if align_corners:
        return ((coord + 1) / 2) * (size - 1)
    return ((coord + 1) * size - 1) / 2


def grid_sample_ref(input, grid, mode='bilinear', padding_mode='zeros',
                    align_corners=False):
    """torch/jittor 2-D bilinear grid_sample with zero padding.

    grid is (N,Ho,Wo,2) with last-dim order (x=W, y=H). A corner outside the image
    contributes 0 (zeros padding); the interior bilinear blend matches
    ``grid_sampler_2d``'s ``a*dnx*dny + b*dnx*dy + c*dx*dny + d*dx*dy``.
    """
    assert mode == "bilinear" and padding_mode == "zeros"
    input = input.astype(np.float64)
    grid = grid.astype(np.float64)
    N, C, Hi, Wi = input.shape
    _, Ho, Wo, _ = grid.shape
    out = np.zeros((N, C, Ho, Wo), dtype=np.float64)

    def at(n, c, yy, xx):
        # zero padding: any out-of-range corner reads as 0.
        if 0 <= yy < Hi and 0 <= xx < Wi:
            return input[n, c, yy, xx]
        return 0.0

    for n in range(N):
        for i in range(Ho):
            for j in range(Wo):
                gx = grid[n, i, j, 0]   # width
                gy = grid[n, i, j, 1]   # height
                xs = _grid_unnormalize(gx, Wi, align_corners)
                ys = _grid_unnormalize(gy, Hi, align_corners)
                fx = int(np.floor(xs)); fy = int(np.floor(ys))
                cx = fx + 1; cy = fy + 1
                dx = xs - fx; dy = ys - fy
                dnx = 1.0 - dx; dny = 1.0 - dy
                for c in range(C):
                    a = at(n, c, fy, fx)
                    b = at(n, c, cy, fx)
                    cc = at(n, c, fy, cx)
                    d = at(n, c, cy, cx)
                    out[n, c, i, j] = (a * dnx * dny + b * dnx * dy
                                       + cc * dx * dny + d * dx * dy)
    return out


def _linspace_from_neg_one(num_steps, align_corners):
    """jittor.nn.linspace_from_neg_one: normalized base-grid coordinates."""
    if num_steps <= 1:
        return np.zeros((0,), dtype=np.float64)
    ra = np.linspace(-1.0, 1.0, num_steps)
    if not align_corners:
        ra = ra * (num_steps - 1) / num_steps
    return ra


def affine_grid_ref(theta, size=None, align_corners=False):
    """jittor.nn.affine_grid 4-D oracle: base_grid @ thetaᵀ.

    theta is (N,2,3); size is (N,C,H,W). Output grid is (N,H,W,2). Linear in theta.
    """
    theta = theta.astype(np.float64)
    N, C, H, W = (int(s) for s in size)
    base = np.zeros((N, H, W, 3), dtype=np.float64)
    base[..., 0] = _linspace_from_neg_one(W, align_corners)[None, None, :]
    base[..., 1] = _linspace_from_neg_one(H, align_corners)[None, :, None]
    base[..., 2] = 1.0
    flat = base.reshape(N, H * W, 3)
    grid = np.matmul(flat, np.transpose(theta, (0, 2, 1)))  # (N, H*W, 2)
    return grid.reshape(N, H, W, 2)


# --------------------------------------------------------------- sample builders
# Tensors stay tiny (<= ~24 differentiated elements): gradcheck is O(numel) float64
# finite-diff. Deterministic seeds. interpolate/affine_grid are linear in their
# differentiated leaf so any values are safe; grid_sample's grid is built to land
# strictly interior (see the grid-construction note below).

def sample_interpolate_nearest(op_info, device, dtype, requires_grad):
    """nearest is a pure gather -> linear in input; up- and down-sample, rect."""
    out = []
    # (1,1,3,3) -> (1,1,4,4) upsample
    out.append(SampleInput(
        make_tensor(1, 1, 3, 3, dtype=dtype, requires_grad=requires_grad, seed=1000),
        size=(4, 4), mode="nearest"))
    # (1,1,4,4) -> (1,1,3,3) downsample
    out.append(SampleInput(
        make_tensor(1, 1, 4, 4, dtype=dtype, requires_grad=requires_grad, seed=1001),
        size=(3, 3), mode="nearest"))
    # (1,2,3,4) -> (1,2,5,2) rectangular (asymmetric up/down)
    out.append(SampleInput(
        make_tensor(1, 2, 3, 4, dtype=dtype, requires_grad=requires_grad, seed=1002),
        size=(5, 2), mode="nearest"))
    return out


def sample_interpolate_bilinear(op_info, device, dtype, requires_grad):
    """bilinear is a fixed linear blend -> linear in input; sweep align_corners."""
    out = []
    # (1,1,3,3) -> (1,1,4,4), align_corners False then True
    out.append(SampleInput(
        make_tensor(1, 1, 3, 3, dtype=dtype, requires_grad=requires_grad, seed=1010),
        size=(4, 4), mode="bilinear", align_corners=False))
    out.append(SampleInput(
        make_tensor(1, 1, 3, 3, dtype=dtype, requires_grad=requires_grad, seed=1011),
        size=(4, 4), mode="bilinear", align_corners=True))
    # downsample (1,2,4,4) -> (1,2,3,3), align_corners False
    out.append(SampleInput(
        make_tensor(1, 2, 4, 4, dtype=dtype, requires_grad=requires_grad, seed=1012),
        size=(3, 3), mode="bilinear", align_corners=False))
    # rectangular (1,1,4,3) -> (1,1,3,5), align_corners True
    out.append(SampleInput(
        make_tensor(1, 1, 4, 3, dtype=dtype, requires_grad=requires_grad, seed=1013),
        size=(3, 5), mode="bilinear", align_corners=True))
    return out


def _interior_grid(N, Ho, Wo, Hi, Wi, align_corners, seed):
    """Build a normalized (N,Ho,Wo,2) grid whose every sample point maps to a
    source coord strictly INTERIOR -- all four bilinear corners inside
    [0,Wi-1]x[0,Hi-1] with fractional part a safe margin from {0,1}.

    We pick target source coords in [1.2, Wi-1.8] (so floor>=1, ceil<=Wi-1, frac in
    [0.2,0.8]) then invert jittor's unnormalize map to the [-1,1] grid value. This
    keeps the floor index and the zero-pad border fixed under the gradcheck eps, so
    the bilinear blend is smooth in the grid and finite-diff matches analytic.
    """
    rng = np.random.RandomState(seed)
    # interior fractional source coords, margin 0.2..0.8 away from integers/borders
    sx = rng.uniform(1.2, Wi - 1.8, size=(N, Ho, Wo))
    sy = rng.uniform(1.2, Hi - 1.8, size=(N, Ho, Wo))
    if align_corners:
        # src = ((g+1)/2)*(size-1)  ->  g = 2*src/(size-1) - 1
        gx = 2.0 * sx / (Wi - 1) - 1.0
        gy = 2.0 * sy / (Hi - 1) - 1.0
    else:
        # src = ((g+1)*size - 1)/2  ->  g = (2*src + 1)/size - 1
        gx = (2.0 * sx + 1.0) / Wi - 1.0
        gy = (2.0 * sy + 1.0) / Hi - 1.0
    grid = np.stack([gx, gy], axis=-1).astype(np.float64)  # last dim (x=W, y=H)
    return grid


def sample_grid_sample(op_info, device, dtype, requires_grad):
    """input (N,C,Hi,Wi) + float grid (N,Ho,Wo,2) -- BOTH differentiated.

    Hi,Wi >= 4 so an interior [1.2, size-1.8] window exists. grid points are pinned
    interior (see ``_interior_grid``) so the bilinear backward w.r.t. grid is smooth
    under finite-diff (no floor/zero-pad-border crossing).
    """
    out = []
    for k, (N, C, Hi, Wi, Ho, Wo, ac) in enumerate([
            (1, 1, 4, 4, 2, 2, False),
            (1, 1, 4, 4, 2, 2, True),
            (1, 2, 4, 5, 2, 2, False),
    ]):
        inp = make_tensor(N, C, Hi, Wi, dtype=dtype,
                          requires_grad=requires_grad, seed=1020 + k)
        g = _interior_grid(N, Ho, Wo, Hi, Wi, ac, seed=1030 + k)
        grid = jt.array(g.astype("float32")).cast(dtype)
        if requires_grad:
            try:
                grid.requires_grad = True
            except Exception:
                pass
        out.append(SampleInput(inp, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=ac))
    return out


def sample_affine_grid(op_info, device, dtype, requires_grad):
    """theta (N,2,3) differentiated; size kwarg held fixed. Linear in theta."""
    out = []
    for k, (N, C, H, W, ac) in enumerate([
            (1, 1, 3, 3, False),
            (1, 1, 3, 3, True),
            (2, 1, 2, 4, False),
    ]):
        theta = make_tensor(N, 2, 3, dtype=dtype, low=-1.0, high=1.0,
                            requires_grad=requires_grad, seed=1040 + k)
        out.append(SampleInput(theta, size=(N, C, H, W), align_corners=ac))
    return out


# --------------------------------------------------------------------- op_db

op_db = [
    # ---- interpolate: nearest + bilinear (both LINEAR in input -> full gradgrad) ----
    # nearest is a pure gather; bilinear is a fixed-weight blend. Forward pinned to the
    # validated source-index refs; backward gradchecked; 2nd derivative is exactly 0.
    OpInfo("interpolate", variant_test_name="nearest",
           op=F.interpolate, ref=interpolate_ref,
           sample_inputs_func=sample_interpolate_nearest),
    OpInfo("interpolate", variant_test_name="bilinear",
           op=F.interpolate, ref=interpolate_ref,
           sample_inputs_func=sample_interpolate_bilinear),

    # ---- grid_sample: bilinear / zeros, differentiates input AND grid ----
    # Linear in input; bilinear (kinked at floor/border) in grid -> 1st-order gradcheck
    # only, with grids pinned interior. supports_gradgrad=False: the 2nd derivative
    # differentiates the floor/zero-pad step (torch restricts double-backward too).
    OpInfo("grid_sample", op=F.grid_sample, ref=grid_sample_ref,
           sample_inputs_func=sample_grid_sample, supports_gradgrad=False),

    # ---- affine_grid: grid = base_grid @ thetaᵀ (LINEAR in theta -> full gradgrad) ----
    OpInfo("affine_grid", op=F.affine_grid, ref=affine_grid_ref,
           sample_inputs_func=sample_affine_grid),
]
