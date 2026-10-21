"""Floating floor_divide / mod / fmod follow IEEE and NumPy (jittor-core-gaps.md §3.6).

Floats used to compute floor(x / y) and x - floor(x / y) * y from the rounded
quotient: 1 // 0.1 was 10 (the exact quotient is just below 10), the remainder
of a finite x over an infinite y was 0 * inf = NaN, and a zero remainder lost
the divisor's sign. They now follow NumPy's divmod, which starts from the
exact fmod remainder; the native `fmod` is C fmod. Every value is compared
bitwise with NumPy, so the sign of a zero counts.
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


_PAIRS = [
    # infinite divisors
    (1.0, np.inf), (-1.0, np.inf), (1.0, -np.inf), (-1.0, -np.inf), (np.inf, 2.0),
    # signed zeros
    (-0.0, 1.0), (0.0, -1.0), (-0.0, -1.0), (0.0, 1.0), (-1.0, 1.0), (1.0, -1.0),
    # decimal divisors: the exact quotient rounds up to an integer
    (1.0, 0.1), (-1.0, 0.1), (1.0, -0.1), (-1.0, -0.1), (0.3, 0.1),
    # ordinary values, NaN, zero divisors, a huge quotient
    (5.5, 2.0), (-5.5, 2.0), (-2.7, 2.0), (np.nan, 1.0), (1.0, np.nan),
    (7.0, 0.0), (-7.0, 0.0), (0.0, 0.0), (1e300, 1e-300),
]


def _bitwise_equal(got, want):
    got, want = np.asarray(got), np.asarray(want)
    same = (got == want) & (np.signbit(got) == np.signbit(want))
    return same | (np.isnan(got) & np.isnan(want))


_OPS = {
    "floor_divide": (lambda x, y: jt.floor_divide(x, y), np.floor_divide),
    "mod": (lambda x, y: x % y, np.mod),
    "fmod": (lambda x, y: jt.fmod(x, y), np.fmod),
    "Var.fmod": (lambda x, y: x.fmod(y), np.fmod),
}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_OPS))
def test_edge_values_match_numpy_bitwise(device, name, dtype):
    op, ref = _OPS[name]
    x_np = np.array([p[0] for p in _PAIRS], dtype=dtype)
    y_np = np.array([p[1] for p in _PAIRS], dtype=dtype)
    got = op(jt.array(x_np, dtype=dtype), jt.array(y_np, dtype=dtype))
    got.sync(device_sync=True, weak_sync=False)
    assert got.location() == ("device" if device == "cuda" else "cpu")
    got = got.numpy()
    with np.errstate(all="ignore"):
        want = ref(x_np, y_np)
    bad = [(_PAIRS[i], got[i], want[i]) for i in np.nonzero(~_bitwise_equal(got, want))[0]]
    assert not bad, bad


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("name", ["floor_divide", "mod", "fmod"])
def test_half_precision_rounds_the_float32_result(device, name, dtype):
    op, ref = _OPS[name]
    x_np = np.array([5.5, -5.5, 1.0, -1.0, 7.25, -0.0], dtype=np.float32)
    y_np = np.array([2.0, 2.0, 0.5, 0.5, -2.0, 1.0], dtype=np.float32)
    got = op(jt.array(x_np).cast(dtype), jt.array(y_np).cast(dtype))
    assert str(got.dtype) == dtype
    got = got.float32().numpy()
    want = ref(x_np, y_np)       # exactly representable in both half formats
    assert _bitwise_equal(got, want).all(), (got, want)


def test_integer_fmod_takes_the_dividends_sign(device):
    x = np.array([7, -7, 7, -7, 0], dtype=np.int32)
    y = np.array([3, 3, -3, -3, 5], dtype=np.int32)
    got = jt.fmod(jt.array(x), jt.array(y))
    assert str(got.dtype) == "int32"
    np.testing.assert_array_equal(got.numpy(), np.fmod(x, y))
    np.testing.assert_array_equal((jt.array(x) % jt.array(y)).numpy(), np.mod(x, y))


@pytest.mark.parametrize("name", ["mod", "fmod"])
def test_both_gradients_match_finite_differences(device, name):
    op, ref = _OPS[name]
    # Away from the discontinuities (1.0 over 0.1 sits right at one: its
    # remainder is just below 0.1), including decimal divisors.
    x_np = np.array([5.5, -5.5, 1.03, -1.3, 0.75, 2.0], dtype=np.float64)
    y_np = np.array([2.0, 2.0, 0.1, 0.4, -0.5, -0.3], dtype=np.float64)
    w_np = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 0.25])
    x = jt.array(x_np, dtype="float64")
    y = jt.array(y_np, dtype="float64")
    loss = (op(x, y) * jt.array(w_np, dtype="float64")).sum()
    gx, gy = (g.numpy() for g in jt.grad(loss, [x, y]))
    eps = 1e-7
    f = lambda a, b: np.sum(ref(a, b) * w_np)
    fd_x = np.array([(f(x_np + eps * e, y_np) - f(x_np - eps * e, y_np)) / (2 * eps)
                     for e in np.eye(6)])
    fd_y = np.array([(f(x_np, y_np + eps * e) - f(x_np, y_np - eps * e)) / (2 * eps)
                     for e in np.eye(6)])
    np.testing.assert_allclose(gx, fd_x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy, fd_y, rtol=1e-6, atol=1e-6)
