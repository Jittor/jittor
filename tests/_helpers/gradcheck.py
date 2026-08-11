# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``gradcheck`` / ``gradgradcheck`` -- the backward-correctness oracle.

Faithful jittor port of ``torch.autograd.gradcheck``: it compares the **analytical**
Jacobian (assembled from ``jt.grad``) against the **numerical** Jacobian (central
finite differences of the forward) and asserts they agree. This is how PyTorch
verifies every operator's derivative, and it is the right tool here too:

  * It needs no second framework and no stored "golden" values.
  * Combined with a forward pinned to an independent numpy reference (``OpInfo.ref``),
    it transitively proves **torch-parity of the backward**: jittor's analytical grad
    must equal the numerical derivative of a forward that already equals numpy/torch.
  * It catches the wrong-formula silent-wrong bugs this project is full of -- a
    stubbed ``solve`` grad that returns 0, an svd grad contracting the wrong axis, a
    scatter backward that drops contributions -- because those are wrong in exact
    arithmetic, so the analytical Jacobian diverges from finite differences.

As in torch, float inputs are upcast to float64 before differencing (float32 finite
differences are too noisy to certify a gradient). A float32-specific numerical
stability bug -- e.g. the normalization small-variance cancellation -- is therefore
NOT this tool's job; those get dedicated float32 precision tests. gradcheck owns
"is the derivative formula correct".
"""
import numpy as np
import jittor as jt

from .common import to_numpy


class GradcheckError(AssertionError):
    """Raised when the analytical and numerical Jacobians disagree."""


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _float64_inputs(inputs):
    arrs = []
    for x in inputs:
        a = to_numpy(x)
        if a.dtype.kind == "f":
            a = a.astype("float64")
        arrs.append(a)
    return arrs


def _call(func, arrays):
    """Build fresh jittor leaf Vars from arrays, run func, return (vars, outputs).

    NB: ``jt.array(float64_ndarray)`` silently narrows to float32 (the torch_compat
    default), which would defeat the float64 finite differences. We pin the dtype
    explicitly so the FD step actually runs in double precision.
    """
    vs = []
    for a in arrays:
        a = np.ascontiguousarray(a)
        vs.append(jt.array(a, dtype=str(a.dtype)))
    out = func(*vs)
    return vs, _as_list(out)


def _analytical_jacobians(vs, outputs, n_in_elems):
    """For each (output, input) build the analytical Jacobian via ``jt.grad``.

    Row ``k`` of J[output o][input i] is d(o_flat[k]) / d(input_i), obtained by
    differentiating each scalar output element back to every input.
    """
    jac = []  # jac[o][i] -> ndarray (out_o.numel, in_i.numel)
    for o in outputs:
        of = o.reshape(-1)
        O = of.shape[0]
        per_in = [np.zeros((O, s), dtype="float64") for s in n_in_elems]
        for k in range(O):
            grads = jt.grad(of[k], vs, retain_graph=True)
            for i, g in enumerate(grads):
                gi = to_numpy(g).reshape(-1).astype("float64")
                # jt.grad returns a zero Var of the input shape when disconnected
                per_in[i][k, :] = gi
        jac.append(per_in)
    return jac


def _numerical_jacobians(func, base, outputs, eps):
    """For each (output, input) build the numerical Jacobian by central differences.

    Column ``j`` of J[output o][input i] is (o(x + eps e_j) - o(x - eps e_j)) / 2eps.
    """
    n_out = len(outputs)
    out_sizes = [to_numpy(o).size for o in outputs]
    n_in_elems = [a.size for a in base]
    jac = [[np.zeros((out_sizes[o], n_in_elems[i]), dtype="float64")
            for i in range(len(base))] for o in range(n_out)]
    for i in range(len(base)):
        flat = base[i].reshape(-1)
        for j in range(flat.size):
            orig = float(flat[j])
            flat[j] = orig + eps
            _, op = _call(func, base)
            flat[j] = orig - eps
            _, om = _call(func, base)
            flat[j] = orig
            for o in range(n_out):
                yp = to_numpy(op[o]).reshape(-1).astype("float64")
                ym = to_numpy(om[o]).reshape(-1).astype("float64")
                jac[o][i][:, j] = (yp - ym) / (2 * eps)
    return jac


def gradcheck(func, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3,
              raise_exception=True, nondet_tol=0.0):
    """Check that ``func``'s analytical gradient matches finite differences.

    Args:
      func:   callable taking the differentiable inputs (as jittor Vars) and
              returning a Var or sequence of Vars. Non-differentiable arguments
              (indices, flags) should be closed over, not passed here.
      inputs: a Var or list of Vars to differentiate with respect to.
      eps:    finite-difference step (on float64-upcast inputs).
      atol/rtol: Jacobian agreement tolerance (torch defaults).
      nondet_tol: extra absolute slack for nondeterministic ops (default 0).

    Returns ``True`` if all Jacobians agree; otherwise raises
    :class:`GradcheckError` (or returns ``False`` if ``raise_exception=False``).
    """
    inputs = _as_list(inputs)
    base = _float64_inputs(inputs)
    n_in_elems = [a.size for a in base]

    vs, outputs = _call(func, base)
    outputs = [o for o in outputs if to_numpy(o).dtype.kind == "f"]
    if not outputs:
        raise GradcheckError("gradcheck: func produced no floating-point output")

    analytical = _analytical_jacobians(vs, outputs, n_in_elems)
    numerical = _numerical_jacobians(func, base, outputs, eps)

    failures = []
    for o in range(len(outputs)):
        for i in range(len(base)):
            a = analytical[o][i]
            n = numerical[o][i]
            if not np.allclose(a, n, atol=atol + nondet_tol, rtol=rtol):
                diff = np.abs(a - n)
                k = int(np.unravel_index(np.argmax(diff), diff.shape)[0])
                failures.append(
                    f"Jacobian mismatch output[{o}] wrt input[{i}]: "
                    f"max|analytical-numerical|={diff.max():.3e} "
                    f"(analytical[{k}]={a.flat[int(np.argmax(diff))]:.6g}, "
                    f"numerical={n.flat[int(np.argmax(diff))]:.6g})")
    if failures:
        if raise_exception:
            raise GradcheckError("\n".join(failures))
        return False
    return True


def numerical_vjp(func, inputs, cotangents, *, eps=1e-6):
    """High-accuracy numerical gradient of ``sum(cot . func(x))`` w.r.t. each input.

    Computes the vector-Jacobian product by central differences on **float64**-upcast
    inputs, regardless of the dtype the analytical gradient was taken in. This is the
    "true gradient" oracle used by the float32 precision-stability tests (e.g.
    normalization small-variance backward): the analytical float32 gradient must
    match this float64 reference, or a catastrophic-cancellation backward is exposed.
    """
    inputs = _as_list(inputs)
    cots = [to_numpy(c).astype("float64").reshape(-1) for c in _as_list(cotangents)]
    base = _float64_inputs(inputs)
    grads = [np.zeros_like(a) for a in base]
    for i in range(len(base)):
        flat = base[i].reshape(-1)
        gflat = grads[i].reshape(-1)
        for j in range(flat.size):
            orig = float(flat[j])
            flat[j] = orig + eps
            _, op = _call(func, base)
            flat[j] = orig - eps
            _, om = _call(func, base)
            flat[j] = orig
            total = 0.0
            for o_idx in range(len(op)):
                yp = to_numpy(op[o_idx]).reshape(-1).astype("float64")
                ym = to_numpy(om[o_idx]).reshape(-1).astype("float64")
                total += float(np.dot(cots[o_idx], (yp - ym) / (2 * eps)))
            gflat[j] = total
    return grads


def gradgradcheck(func, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3,
                  raise_exception=True, nondet_tol=0.0):
    """Check second derivatives: gradcheck applied to the first-order gradient.

    Forms ``g(x) = d/dx sum(w * func(x))`` for a fixed random ``w`` (so the
    first-order grad is itself a differentiable function of ``x``), then gradchecks
    ``g``. Verifies that ``jt.grad`` is itself differentiable -- the property the
    create_graph / higher-order autograd path relies on.
    """
    inputs = _as_list(inputs)
    base = _float64_inputs(inputs)
    # fixed cotangents per output, drawn once so g is deterministic
    vs0, outs0 = _call(func, base)
    rng = np.random.RandomState(0)
    weights = [rng.randn(*to_numpy(o).shape).astype("float64") for o in outs0]

    def grad_fn(*vars_):
        out = _as_list(func(*vars_))
        loss = None
        for o, w in zip(out, weights):
            term = (o * jt.array(w)).sum()
            loss = term if loss is None else loss + term
        gs = jt.grad(loss, list(vars_), retain_graph=True)
        # return the concatenation of grads as the "output" to be re-differentiated
        return gs[0] if len(gs) == 1 else tuple(gs)

    return gradcheck(grad_fn, inputs, eps=eps, atol=atol, rtol=rtol,
                     raise_exception=raise_exception, nondet_tol=nondet_tol)
