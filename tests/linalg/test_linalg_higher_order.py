"""Second-order AD through linalg, real and native complex (jittor-core-gaps.md §3.5).

eigh, cholesky, qr and svd used to take their backward from an opaque
numpy_code callback (native complex went through the ComplexNumber callback
chain), so a second derivative raised "numpy_code has no backward callback".
Their adjoints are now written with differentiable ops around a live
recursive call, like inv/solve/det. Each case checks the first-order gradient
against central finite differences of the loss, and the Hessian-vector product
against central finite differences of that gradient, on every device.

Hermitian inputs are parametrised as sym(X) + shift*I so a perturbation of X
stays in the domain eigh/cholesky actually read; eigenvector and singular
vector losses are invariant to the per-column phase (sign) gauge.

Also pins first-order complex gradients that were silently wrong: det
(missing conjugates), cholesky (real-only transposes) and slogdet (missing
conjugate and sign term, covered by its case below).
"""

import numpy as np
import pytest
import jittor as jt

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


def _var(x):
    return jt.array(np.asarray(x), dtype="complex128" if np.iscomplexobj(x) else "float64")


def _is_complex(v):
    return "complex" in str(v.dtype)


def _smooth(y, k=1.0):
    if _is_complex(y):
        return (y.real * (1.3 * k)).sin().sum() + (y.imag * (0.7 * k)).cos().sum()
    return (y * (1.1 * k)).sin().sum()


def _gauge_invariant(y, k=1.0):
    absq = (y * y.conj()).real if _is_complex(y) else y * y
    return (absq * (0.9 * k)).sin().sum()


def _herm(x, shift):
    n = x.shape[-1]
    eye = jt.array(np.eye(n), dtype="float64")
    h = (x + x.transpose(-1, -2).conj()) * 0.5
    return h + (eye * (shift + 0j) if _is_complex(x) else eye * shift)


def _real_part(y):
    return y.real if _is_complex(y) else y


# Built on call, not at import: the lambdas name local helpers, and a module
# that runs nothing at collection is a structure-gate contract.
def _cases():
    return {
        "eigh values": (lambda x: _real_part(jt.linalg.eigh(_herm(x, 0.0))[0]).sin().sum(), (3, 3)),
        "eigh vectors": (lambda x: _gauge_invariant(jt.linalg.eigh(_herm(x, 0.0))[1]), (3, 3)),
        "cholesky": (lambda x: _smooth(jt.linalg.cholesky(_herm(x, 8.0))), (3, 3)),
        "qr tall": (lambda x: _smooth(jt.linalg.qr(x)[0]) + _smooth(jt.linalg.qr(x)[1], 0.5), (4, 2)),
        "qr wide": (lambda x: _smooth(jt.linalg.qr(x)[0]) + _smooth(jt.linalg.qr(x)[1], 0.5), (2, 4)),
        "svd values": (lambda x: _real_part(jt.linalg.svd(x)[1]).sin().sum(), (3, 3)),
        "svd vectors tall": (lambda x: _gauge_invariant(jt.linalg.svd(x)[0])
                             + _gauge_invariant(jt.linalg.svd(x)[2], 0.6), (4, 2)),
        "svd vectors wide": (lambda x: _gauge_invariant(jt.linalg.svd(x)[0])
                             + _gauge_invariant(jt.linalg.svd(x)[2], 0.6), (2, 4)),
        "svd reconstruction": (lambda x: _smooth(jt.matmul(
            jt.linalg.svd(x)[0] * jt.linalg.svd(x)[1].unsqueeze(-2), jt.linalg.svd(x)[2])), (3, 2)),
        "inv": (lambda x: _smooth(jt.linalg.inv(x)), (3, 3)),
        "det": (lambda x: _smooth(jt.linalg.det(x)), (3, 3)),
        # sign and log|det|: the complex sign term was missing from the gradient.
        "slogdet": (lambda x: _smooth(jt.linalg.slogdet(x)[0], 0.8)
                    + _real_part(jt.linalg.slogdet(x)[1]).sin().sum(), (3, 3)),
    }


_CASES_NAMES = (
    "eigh values",
    "eigh vectors",
    "cholesky",
    "qr tall",
    "qr wide",
    "svd values",
    "svd vectors tall",
    "svd vectors wide",
    "svd reconstruction",
    "inv",
    "det",
    "slogdet",
)


def _grad(f, x_np):
    x = _var(x_np)
    return jt.grad(f(x), x).numpy()


def _fd_grad(f, x_np, eps=1e-6):
    out = np.zeros_like(x_np)
    for idx in np.ndindex(*x_np.shape):
        for unit in ((1.0, 1j) if np.iscomplexobj(x_np) else (1.0,)):
            d = np.zeros_like(x_np)
            d[idx] = unit * eps
            val = (float(f(_var(x_np + d)).numpy()) - float(f(_var(x_np - d)).numpy())) / (2 * eps)
            out[idx] += val if unit == 1.0 else 1j * val
    return out


def _hvp(f, x_np, v_np):
    x, v = _var(x_np), _var(v_np)
    g = jt.grad(f(x), x)
    inner = (g.real * v.real + g.imag * v.imag).sum() if _is_complex(g) else (g * v).sum()
    return jt.grad(inner, x).numpy()


def _rel(a, b):
    return np.abs(a - b).max() / max(1.0, np.abs(b).max())


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("name", _CASES_NAMES)
def test_first_and_second_order_match_finite_differences(device, name, kind):
    f, shape = _cases()[name]
    rng = np.random.default_rng(abs(hash((name, kind))) % (2 ** 32))
    cplx = kind == "complex"
    x = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
    x = x + 2 * np.eye(*shape)          # well separated singular values / pivots
    v = rng.standard_normal(shape) + (1j * rng.standard_normal(shape) if cplx else 0)
    assert _rel(_grad(f, x), _fd_grad(f, x)) < 1e-6
    eps = 1e-5
    fd_hvp = (_grad(f, x + eps * v) - _grad(f, x - eps * v)) / (2 * eps)
    assert _rel(_hvp(f, x, v), fd_hvp) < 1e-5


def test_complex_det_first_order_uses_conjugates(device):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)) + 3 * np.eye(3)
    w = rng.standard_normal(()) + 1j * rng.standard_normal(())
    x = _var(a)
    # Re(conj(w) det(A)): its conjugate-Wirtinger gradient is w * conj(det A) A^-H.
    # The weight is a complex128 Var: Python scalars enter float math at
    # single precision (a separate issue), which would dominate the error.
    loss = (jt.linalg.det(x) * _var(np.array(np.conj(w)))).real.sum()
    expected = w * np.conj(np.linalg.det(a)) * np.linalg.inv(a).conj().T
    np.testing.assert_allclose(jt.grad(loss, x).numpy(), expected, rtol=1e-9, atol=1e-10)


def test_complex_cholesky_first_order_is_hermitian_adjoint(device):
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    w_np = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    d_np = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    a_np = (x_np + x_np.conj().T) / 2 + 8 * np.eye(3)
    d_np = (d_np + d_np.conj().T) / 2          # a Hermitian direction
    a = _var(a_np)
    loss = (jt.linalg.cholesky(a) * _var(w_np.conj())).real.sum()   # Re<W, L>
    g = jt.grad(loss, a).numpy()
    f = lambda m: np.real(np.sum(np.conj(w_np) * np.linalg.cholesky(m)))
    eps = 1e-6
    fd = (f(a_np + eps * d_np) - f(a_np - eps * d_np)) / (2 * eps)
    np.testing.assert_allclose(np.real(np.sum(np.conj(g) * d_np)), fd, rtol=1e-7)


def _herm_shifted(x):
    return _herm(x, 6.0)


# Built on call, not at import: the lambdas name local helpers, and a module
# that runs nothing at collection is a structure-gate contract.
def _composed():
    return {
        "svd values": (lambda x: jt.linalg.svd(x)[1], lambda a: np.linalg.svd(a)[1]),
        "eigh values": (lambda x: jt.linalg.eigh(_herm_shifted(x))[0],
                        lambda a: np.linalg.eigh((a + a.T) / 2 + 6 * np.eye(3))[0]),
        "cholesky": (lambda x: jt.linalg.cholesky(_herm_shifted(x)),
                     lambda a: np.linalg.cholesky((a + a.T) / 2 + 6 * np.eye(3))),
        "qr R": (lambda x: jt.linalg.qr(x)[1], lambda a: np.linalg.qr(a)[1]),
    }


_COMPOSED_NAMES = (
    "svd values",
    "eigh values",
    "cholesky",
    "qr R",
)


@pytest.mark.parametrize("name", _COMPOSED_NAMES)
def test_jvp_vjp_and_vmap_of_grad(device, name):
    from jittor.autograd.functional import jvp, vjp
    f_jt, f_np = _composed()[name]
    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, 3)) + 2 * np.eye(3)
    t = rng.standard_normal((3, 3))
    eps = 1e-6
    fd = (f_np(a + eps * t) - f_np(a - eps * t)) / (2 * eps)
    _, tangent = jvp(f_jt, _var(a), _var(t))
    np.testing.assert_allclose(tangent.numpy(), fd, atol=1e-6)
    cotangent = rng.standard_normal(np.shape(fd))
    _, pulled = vjp(f_jt, _var(a), _var(cotangent))
    np.testing.assert_allclose(np.sum(pulled.numpy() * t), np.sum(cotangent * fd), atol=1e-6)
    batch = np.stack([a, a + 0.1 * t, a - 0.2 * t])
    mapped = jt.vmap(lambda x: jt.grad(f_jt(x).sum(), x))(_var(batch)).numpy()
    looped = []
    for b in batch:
        x = _var(b)
        looped.append(jt.grad(f_jt(x).sum(), x).numpy())
    np.testing.assert_allclose(mapped, np.stack(looped), rtol=1e-12, atol=1e-12)
