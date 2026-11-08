# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# SVD and eigh gradients at repeated and zero singular values / eigenvalues.
#
# The vector adjoints divide by the gaps s_j^2 - s_i^2 (w_j - w_i for eigh),
# and by S itself for the complex phase term and the projection terms of a
# non-square SVD. A repeated or zero value made that 1/0, and the 0 * inf
# that followed turned the whole gradient into NaN with no warning, even for
# a loss that does not depend on the basis of the repeated subspace -- every
# gradient of an MPS simulation, which starts from a product state. Such a
# loss has a well-defined gradient, and it is what these tests check against.
#
# Ground truth: loss = ||U_2 S_2 Vh_2||_F^2 = s_1^2 + s_2^2, whose gradient
# is 2 * A_2 (A_2 the rank-2 truncation), so long as s_2 > s_3. For eigh,
# loss = tr(P A) with P the projector onto the top-2 eigenspace has gradient
# P^T (folded onto the lower triangle for complex input); a degenerate pair
# and a split pair spanning the same subspace give the same gradient.
import numpy as np
import pytest
import jittor as jt
from jittor import linalg

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def _unitary(rng, n, complex_):
    z = rng.normal(size=(n, n))
    if complex_:
        z = z + 1j * rng.normal(size=(n, n))
    return np.linalg.qr(z)[0]


def _top2_svd_loss_grad(a_np, dtype):
    a = jt.array(a_np.astype(dtype))
    u, s, vh = linalg.svd(a)
    loss = (jt.matmul(u[:, :2] * s[:2].unsqueeze(-2), vh[:2, :])).abs().sqr().sum()
    return jt.grad(loss, a)


_CASES = {
    "distinct": [4.0, 3.0, 2.0, 1.0],
    "repeated": [2.0, 2.0, 1.0, 0.5],
    "zero": [3.0, 2.0, 0.0, 0.0],
}
_ATOL = {"float32": 1e-4, "complex64": 1e-4, "complex128": 1e-11}


# (m, n) = (4, 6) and (6, 4) add the (I - U U^H) gU S^-1 and
# S^-1 gVh (I - V V^H) terms, which divide by the zero singular values too.
@pytest.mark.parametrize("shape", [(4, 4), (6, 4), (4, 6)])
@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_svd_gradient_at_repeated_and_zero_singular_values(device, case, dtype, shape):
    m, n = shape
    rng = np.random.default_rng(1)
    complex_ = dtype.startswith("complex")
    u = _unitary(rng, m, complex_)[:, :4]
    v = _unitary(rng, n, complex_)[:, :4]
    s = _CASES[case]
    g = _top2_svd_loss_grad(u @ np.diag(s) @ v.conj().T, dtype).numpy()
    expected = 2 * u[:, :2] @ np.diag(s[:2]) @ v[:, :2].conj().T
    assert np.isfinite(g).all(), g
    np.testing.assert_allclose(g, expected, atol=_ATOL[dtype])


def test_svd_second_derivative_at_repeated_singular_values_is_finite(device):
    rng = np.random.default_rng(3)
    u = _unitary(rng, 4, True)
    a = jt.array((u @ np.diag([2.0, 2.0, 1.0, 0.0]) @ u.conj().T).astype("complex64"))
    u_, s, vh = linalg.svd(a)
    loss = (jt.matmul(u_[:, :2] * s[:2].unsqueeze(-2), vh[:2, :])).abs().sqr().sum()
    g = jt.grad(loss, a)
    gg = jt.grad(g.abs().sqr().sum(), a).numpy()
    assert np.isfinite(gg).all(), gg


def _eigh_projector_loss_grad(h, dtype):
    a = jt.array(h.astype(dtype))
    _, v = linalg.eigh(a)
    p = jt.matmul(v[:, 2:], v[:, 2:].transpose(-1, -2).conj())
    loss = (p * a.transpose(-1, -2)).sum()
    if dtype.startswith("complex"):
        loss = loss.real
    return jt.grad(loss, a).numpy()


@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
def test_eigh_gradient_at_repeated_eigenvalues_matches_a_split_pair(device, dtype):
    rng = np.random.default_rng(2)
    u = _unitary(rng, 4, dtype.startswith("complex"))
    repeated = _eigh_projector_loss_grad(u @ np.diag([0.5, 1.0, 2.0, 2.0]) @ u.conj().T, dtype)
    split = _eigh_projector_loss_grad(u @ np.diag([0.5, 1.0, 1.99, 2.01]) @ u.conj().T, dtype)
    assert np.isfinite(repeated).all(), repeated
    np.testing.assert_allclose(repeated, split, atol=_ATOL[dtype])


# A small gap that is still a gap. diag(1, 1 + 2^-19) in complex64 has a gap
# of 16 eps; the vectors' derivative is C_12 / gap, and a fixed
# `32 * size * eps * max` resolution took it for a repeated value and returned
# a zero gradient and HVP. The resolution is the decomposition's measured
# error now, which a diagonal matrix's exact decomposition leaves at zero.
# Ground truth: the loss Re tr(C P), P the projector onto the top vector
# (phase invariant), differentiated by central differences of numpy's
# complex128 decomposition along random Hermitian directions; the HVP against
# the same input in double precision, where the gap is far from any
# resolution.
def _top_projector_loss_np(h, c, kind):
    if kind == "eigh":
        vec = np.linalg.eigh(h)[1][:, -1]
    else:
        vec = np.linalg.svd(h)[0][:, 0]
    return np.real(np.trace(c @ np.outer(vec, vec.conj())))


def _top_projector_loss(x, c, kind):
    if kind == "eigh":
        vec = linalg.eigh(x)[1][:, -1:]
    else:
        vec = linalg.svd(x)[0][:, :1]
    p = jt.matmul(vec, vec.transpose(-1, -2).conj())
    n = x.shape[-1]
    return (jt.matmul(c, p) * jt.array(np.eye(n), dtype="float32")).real.sum()


def _double(dtype):
    return {"float32": "float64", "complex64": "complex128"}[dtype]


@pytest.mark.parametrize("kind", ["eigh", "svd"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("log2_gap", [-19, -15])
@pytest.mark.parametrize("scale, n", [(1.0, 2), (1e3, 2), (1.0, 4)])
def test_a_small_nonzero_gap_keeps_its_derivative(device, kind, dtype, log2_gap, scale, n):
    rng = np.random.default_rng(abs(log2_gap) * 7 + n)
    cplx = dtype.startswith("complex")
    values = scale * np.array([1.0, 1.0 + 2.0 ** log2_gap, 2.0, 3.0][:n])
    if n == 2:
        values = values[::-1]           # the top vector belongs to the gap
    else:
        values = np.array([scale * 0.5, scale * 0.25, scale * 1.0, scale * (1.0 + 2.0 ** log2_gap)])
    h = np.diag(values).astype(dtype)
    c = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if cplx else 0)
    c = ((c + c.conj().T) / 2).astype(dtype)

    def grad_of(dt):
        x = jt.array(h.astype(dt), dtype=dt)
        return x, jt.grad(_top_projector_loss(x, jt.array(c.astype(dt), dtype=dt), kind), x)

    x, g = grad_of(dtype)
    g_np = g.numpy().astype("complex128")
    gap = scale * 2.0 ** log2_gap
    hh, cc = h.astype("complex128"), c.astype("complex128")
    worst = scale_ref = 0.0
    for _ in range(4):
        d = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if cplx else 0)
        d = (d + d.conj().T) / 2
        t = gap * 1e-4
        fd = (_top_projector_loss_np(hh + t * d, cc, kind)
              - _top_projector_loss_np(hh - t * d, cc, kind)) / (2 * t)
        worst = max(worst, abs(np.real(np.sum(np.conj(g_np) * d)) - fd))
        scale_ref = max(scale_ref, abs(fd))
    assert scale_ref > 1.0 / gap * 1e-3     # the derivative is the large one
    assert worst <= 2e-5 * scale_ref, (worst, scale_ref)

    v = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if cplx else 0)
    v = ((v + v.conj().T) / 2)

    def hvp(dt):
        x, g = grad_of(dt)
        vv = jt.array(v.astype(dt), dtype=dt)
        inner = (g.real * vv.real + g.imag * vv.imag).sum() if cplx else (g * vv).sum()
        return jt.grad(inner, x).numpy().astype("complex128")

    got, expected = hvp(dtype), hvp(_double(dtype))
    assert np.abs(expected).max() > 1.0 / gap ** 2 * 1e-4
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4 * np.abs(expected).max())


# A rotated near-repeated spectrum, I + gap * R / 2 with a 16 eps gap. Jacobi
# at cuSOLVER's default tolerance (CUDA SVD, and CuPy's batched eigh) stopped
# before rotating such a matrix and returned the identity as its vectors: the
# top projector was 0.33 off, the gradient 0.46 and the HVP 1.5. The bounds
# here are what a backward-stable factorization leaves at this gap (torch:
# projector 0.014, gradient 0.15), measured against numpy's complex128
# factorization of the same rounded input.
def _rotated_near_repeat(dtype, offset=0.0):
    eps = np.finfo(np.float32).eps
    gap, theta, phase = 16 * eps, 0.47, 0.63 + offset
    off = np.sin(theta) * (np.exp(1j * phase) if dtype.startswith("complex") else 1.0)
    axis = np.array([[-np.cos(theta), off], [np.conj(off), np.cos(theta)]])
    return (np.eye(2) + gap * axis / 2).astype(dtype), gap


@pytest.mark.parametrize("kind", ["eigh", "svd"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batched", [False, True])
def test_a_rotated_near_repeat_is_resolved(device, kind, dtype, batched):
    rng = np.random.default_rng(31)
    cplx = dtype.startswith("complex")
    members = [_rotated_near_repeat(dtype, offset) for offset in ((0.0, 0.21) if batched else (0.0,))]
    h = np.stack([m for m, _ in members]) if batched else members[0][0]
    gap = members[0][1]
    c = rng.normal(size=(2, 2)) + (1j * rng.normal(size=(2, 2)) if cplx else 0)
    c = ((c + c.conj().T) / 2).astype(dtype)

    x = jt.array(h, dtype=dtype)
    vec = linalg.eigh(x)[1][..., -1:] if kind == "eigh" else linalg.svd(x)[0][..., :1]
    p = jt.matmul(vec, vec.transpose(-1, -2).conj())
    loss = (jt.matmul(jt.array(c, dtype=dtype), p) * jt.array(np.eye(2), dtype="float32")).real.sum()
    g = jt.grad(loss, x).numpy().astype("complex128")
    for b, member in enumerate(h if batched else [h]):
        hh = member.astype("complex128")
        ref = _top_projector_loss_np  # loss of the top projector, complex128
        vec_ref = (np.linalg.eigh(hh)[1][:, -1:] if kind == "eigh" else np.linalg.svd(hh)[0][:, :1])
        p_ref = vec_ref @ vec_ref.conj().T
        got = p.numpy()[b] if batched else p.numpy()
        assert np.linalg.norm(got - p_ref) / np.linalg.norm(p_ref) < 0.05
        gb = g[b] if batched else g
        worst = scale = 0.0
        for _ in range(4):
            d = rng.normal(size=(2, 2)) + (1j * rng.normal(size=(2, 2)) if cplx else 0)
            d = (d + d.conj().T) / 2
            t = gap * 1e-4
            fd = (ref(hh + t * d, c.astype("complex128"), kind)
                  - ref(hh - t * d, c.astype("complex128"), kind)) / (2 * t)
            worst = max(worst, abs(np.real(np.sum(np.conj(gb) * d)) - fd))
            scale = max(scale, abs(fd))
        assert worst <= 0.15 * scale, (worst, scale)
