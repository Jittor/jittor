"""Native vmap: `solve` input combinations and shared-target gradients.

jittor-core-gaps.md §3.4. Every reference is the same computation run per
example with the plain (unbatched) function, so the rule is checked against
the function it batches, not against a second formula.

- `solve`: a mapped matrix with a shared right-hand side (vector or matrix),
  a shared matrix with a mapped right-hand side, logical batch dimensions on
  either side, non-default batch axes, nested levels, pytree outputs, and the
  gradient of every input (a shared input's gradient is the sum over the
  examples).
- `grad` inside vmap with respect to a target the vmapped function does not
  map (a captured parameter): each example's own gradient, not the total.
- A linalg function saved before vmap first installs its patches still
  dispatches to the batching rule.
"""

from pathlib import Path

import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from _helpers.child_process import run_child_script


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


_rng = np.random.default_rng(0)


def _f64(x):
    # jt.array() narrows float64 NumPy input to float32 by default.
    return jt.array(np.asarray(x, dtype=np.float64), dtype="float64")


def _matrices(*shape):
    """Well-conditioned square matrices of shape (*shape, M, M)."""
    m = shape[-1]
    base = _rng.standard_normal(shape[:-1] + (m, m))
    return base + 3 * m * np.eye(m)


def _check(result, expected):
    np.testing.assert_allclose(np.asarray(result.numpy()), expected, rtol=1e-9, atol=1e-10)


def _moved(x, axis):
    return np.moveaxis(x, 0, axis)


# (name, A physical shape, b physical shape, in_dims) -- batch size 3, M = 2.
# `None` marks a shared operand.
_SOLVE_CASES = [
    ("mapped A, shared vector b", (3, 2, 2), (2,), (0, None)),
    ("mapped A, shared matrix b", (3, 2, 2), (2, 1), (0, None)),
    ("mapped A, shared wide b", (3, 2, 2), (2, 4), (0, None)),
    ("shared A, mapped vector b", (2, 2), (3, 2), (None, 0)),
    ("shared A, mapped matrix b", (2, 2), (3, 2, 5), (None, 0)),
    ("both mapped, vector b", (3, 2, 2), (3, 2), (0, 0)),
    ("both mapped, matrix b", (3, 2, 2), (3, 2, 2), (0, 0)),
    ("shared batched A, mapped vector b", (4, 2, 2), (3, 2), (None, 0)),
    ("mapped A, shared batched matrix b", (3, 2, 2), (4, 2, 1), (0, None)),
    ("nondefault axes", (2, 3, 2), (2, 3), (1, 1)),
]


def _example(x, dim, i):
    return x if dim is None else np.take(x, i, axis=dim)


@pytest.mark.parametrize("name, a_shape, b_shape, in_dims", _SOLVE_CASES,
                         ids=[c[0] for c in _SOLVE_CASES])
def test_solve_matches_per_example_solve(device, name, a_shape, b_shape, in_dims):
    a_np = _matrices(*a_shape[:-1]) if in_dims[0] != 1 else _moved(_matrices(3, 2), 1)
    b_np = _rng.standard_normal(b_shape)
    a, b = _f64(a_np), _f64(b_np)
    out = jt.vmap(jt.linalg.solve, in_dims=in_dims)(a, b)
    per_example = [jt.linalg.solve(_f64(_example(a_np, in_dims[0], i)),
                                   _f64(_example(b_np, in_dims[1], i))) for i in range(3)]
    expected = np.stack([p.numpy() for p in per_example])
    # Location before reading: fetching the values may migrate the storage.
    out.sync(device_sync=True, weak_sync=False)
    assert out.location() == ("device" if device == "cuda" else "cpu")
    _check(out, expected)

    # Gradients of every input through the batched solve. A shared input is
    # used by every example, so its gradient is the sum of theirs.
    weights = _rng.standard_normal(expected.shape)
    loss = (out * _f64(weights)).sum()
    ga, gb = jt.grad(loss, [a, b])
    ref_a, ref_b = np.zeros_like(a_np), np.zeros_like(b_np)
    for i in range(3):
        ai = _f64(_example(a_np, in_dims[0], i))
        bi = _f64(_example(b_np, in_dims[1], i))
        li = (jt.linalg.solve(ai, bi) * _f64(weights[i])).sum()
        gai, gbi = (g.numpy() for g in jt.grad(li, [ai, bi]))
        if in_dims[0] is None:
            ref_a += gai
        else:
            np.moveaxis(ref_a, in_dims[0], 0)[i] = gai
        if in_dims[1] is None:
            ref_b += gbi
        else:
            np.moveaxis(ref_b, in_dims[1], 0)[i] = gbi
    _check(ga, ref_a)
    _check(gb, ref_b)


def test_nested_solve_and_pytree_output(device):
    a_np = _matrices(3, 2)                      # (3, 2, 2): outer level
    b_np = _rng.standard_normal((4, 2))         # (4, 2): inner level
    a, b = _f64(a_np), _f64(b_np)

    def per_pair(ai, bj):
        x = jt.linalg.solve(ai, bj)
        return {"x": x, "norm": (x * x).sum()}

    out = jt.vmap(lambda ai: jt.vmap(lambda bj: per_pair(ai, bj))(b))(a)
    ref_x = np.stack([[np.linalg.solve(a_np[i], b_np[j]) for j in range(4)] for i in range(3)])
    _check(out["x"], ref_x)
    _check(out["norm"], (ref_x ** 2).sum(-1))


def test_solve_rhs_rule_is_torch_rule_not_numpy_version():
    # b.shape == a.shape[:-1] is a batch of vectors; (2, 1) against a
    # (2, 2, 2) batch is a batch of 2x1 matrices under every NumPy.
    a_np = _matrices(2, 2)
    b_np = _rng.standard_normal((2, 1))
    out = jt.linalg.solve(_f64(a_np), _f64(b_np))
    _check(out, np.stack([np.linalg.solve(a_np[i], b_np) for i in range(2)]))
    vec = _rng.standard_normal((2, 2))
    out = jt.linalg.solve(_f64(a_np), _f64(vec))
    _check(out, np.stack([np.linalg.solve(a_np[i], vec[i]) for i in range(2)]))


# ---------------------------------------------------------------------------
# grad with respect to targets the vmapped function does not map.
# ---------------------------------------------------------------------------

def test_shared_target_gets_per_example_gradient(device):
    x = _f64([[1.0], [2.0], [3.0]])
    w = _f64([2.0])
    got = jt.vmap(lambda r: jt.grad((r * w).sum(), w))(x)
    _check(got, [[1.0], [2.0], [3.0]])
    # The total is still what differentiating the summed loss outside gives.
    total = jt.grad(jt.vmap(lambda r: (r * w).sum())(x).sum(), w)
    _check(total, [6.0])


def test_mixed_targets_and_nondefault_axis(device):
    x_np = _rng.standard_normal((2, 4))          # batch along axis 1
    w_np = _rng.standard_normal((2,))
    x, w = _f64(x_np), _f64(w_np)

    def f(xi):
        loss = ((xi * w) ** 2).sum() + (xi ** 3).sum()
        return jt.grad(loss, [w, xi])

    gw, gx = jt.vmap(f, in_dims=1)(x)
    ref_w = np.stack([2 * x_np[:, i] ** 2 * w_np for i in range(4)])
    ref_x = np.stack([2 * x_np[:, i] * w_np ** 2 + 3 * x_np[:, i] ** 2 for i in range(4)])
    _check(gw, ref_w)
    _check(gx, ref_x)


def test_nested_levels_partially_shared_target(device):
    # w is mapped by the outer vmap only; the inner level must be separated.
    xs_np = _rng.standard_normal((3, 2))
    ys_np = _rng.standard_normal((4, 2))
    ws_np = _rng.standard_normal((3, 2))
    xs, ys, ws = _f64(xs_np), _f64(ys_np), _f64(ws_np)

    def inner(xi, wi):
        return jt.vmap(lambda yj: jt.grad(((xi + yj) * wi * wi).sum(), wi))(ys)

    got = jt.vmap(inner)(xs, ws)
    ref = np.stack([[2 * (xs_np[i] + ys_np[j]) * ws_np[i] for j in range(4)] for i in range(3)])
    _check(got, ref)


def test_nested_levels_fully_shared_target(device):
    xs_np = _rng.standard_normal((3, 2))
    ys_np = _rng.standard_normal((4, 2))
    w_np = _rng.standard_normal((2,))
    xs, ys, w = _f64(xs_np), _f64(ys_np), _f64(w_np)
    got = jt.vmap(lambda xi: jt.vmap(
        lambda yj: jt.grad((xi * yj * w * w).sum(), w))(ys))(xs)
    ref = np.stack([[2 * xs_np[i] * ys_np[j] * w_np for j in range(4)] for i in range(3)])
    _check(got, ref)


# ---------------------------------------------------------------------------
# A reference saved before vmap installs its patches.
# ---------------------------------------------------------------------------

def _check_saved_reference():
    """Executed only in a fresh interpreter, before any vmap call."""
    import numpy as np
    import jittor as jt
    _f64 = lambda x: jt.array(np.asarray(x, dtype=np.float64), dtype="float64")
    saved_solve, saved_inv = jt.linalg.solve, jt.linalg.inv
    a_np = np.array([[[2.0, 0.0], [0.0, 4.0]], [[1.0, 1.0], [0.0, 2.0]]])
    b_np = np.array([2.0, 4.0])
    out = jt.vmap(lambda a: saved_solve(a, _f64(b_np)))(_f64(a_np))
    np.testing.assert_allclose(out.numpy(), [np.linalg.solve(a, b_np) for a in a_np])
    inv = jt.vmap(lambda a: saved_inv(a))(_f64(a_np))
    np.testing.assert_allclose(inv.numpy(), np.linalg.inv(a_np))
    print("SAVED-REFERENCE-OK", flush=True)


def test_reference_saved_before_install_dispatches():
    child = run_child_script(
        "import runpy\n"
        "runpy.run_path(%r)['_check_saved_reference']()\n" % str(Path(__file__).resolve()),
        name="vmap_saved_reference", text=True, merge_stderr=True,
        without_torch_mode=True)
    assert child.returncode == 0, child.stdout
    assert "SAVED-REFERENCE-OK" in child.stdout
