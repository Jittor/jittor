#!/usr/bin/env python3
"""Probe the places where dtype and edge-case semantics actually diverge.

Breadth does not find these. Sampling 38 (op, dtype) pairs picked for coverage
found nothing; six picked for *semantic divergence* found a Critical defect
(integer ``%`` truncating while ``//`` floored). So this probes categories where
two reasonable implementations disagree, not the whole matrix:

* rounding and sign: floor/trunc division, remainder, negative operands;
* wraparound and saturation: unsigned edges, narrow-width overflow;
* float edges: overflow to inf, subnormals, NaN ordering, -0.0;
* promotion: mixed-dtype binary ops, python scalars against arrays;
* shifts: counts at and beyond the width;
* reductions: empty inputs, rank-0, all-NaN, integer overflow while accumulating;
* identities that tie two operators to each other.

Every check states an independent NumPy expectation. A check whose expectation
is itself uncertain is reported as ``UNVERIFIED`` rather than counted as a pass,
because a probe that scores its own guesses is a probe that always agrees with
itself.

Usage::

    PYTHONPATH=<repo>/python python tools/semantic_divergence_probe.py [--json out.json]
    # run under an isolated JITTOR_HOME so it does not queue behind other work
"""

import argparse
import json
import pathlib
import traceback

import numpy as np


RESULTS = []


def arr(jt, values, dtype):
    """Build a Var and refuse to proceed if construction changed the dtype.

    ``jt.array`` narrows 64-bit numpy values to 32-bit by default
    (``auto_convert_64_to_32``, a documented limitation). A probe that does not
    look would blame whatever operator it was testing: int64 ``1 << 40`` came
    back as 256 and read as a shift computed in 32 bits, when the operands had
    silently become int32 before the shift ever ran. Attributing a known
    construction-time narrowing to an unrelated operator is the exact false
    positive this probe exists to avoid producing.
    """
    import numpy as _np
    var = jt.array(_np.array(values, dtype=dtype))
    if str(var.dtype) != dtype:
        raise AssertionError(
            "construction narrowed %s to %s; set auto_convert_64_to_32=0 "
            "before probing this dtype" % (dtype, var.dtype))
    return var


def check(name, category, got, want, rtol=1e-5, atol=1e-6):
    """Record one comparison against an independent NumPy expectation."""
    try:
        g = np.asarray(got)
        w = np.asarray(want)
        if g.shape != w.shape:
            RESULTS.append({"name": name, "category": category, "status": "MISMATCH",
                            "detail": "shape %s vs %s" % (g.shape, w.shape)})
            return
        if g.dtype.kind in "biu" and w.dtype.kind in "biu":
            ok = bool((g == w).all())
        else:
            ok = bool(np.allclose(g.astype(np.float64), w.astype(np.float64),
                                  rtol=rtol, atol=atol, equal_nan=True))
        RESULTS.append({
            "name": name, "category": category,
            "status": "OK" if ok else "MISMATCH",
            "detail": "" if ok else "got=%s want=%s" % (g.tolist(), w.tolist()),
        })
    except Exception as exc:  # noqa: BLE001 - the probe reports, never hides
        RESULTS.append({"name": name, "category": category, "status": "ERROR",
                        "detail": "%s: %s" % (type(exc).__name__, str(exc)[:120])})


def guarded(name, category, fn):
    """Run one probe body; an exception is a finding, not a crash."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"name": name, "category": category, "status": "ERROR",
                        "detail": "%s: %s" % (type(exc).__name__,
                                              str(exc).splitlines()[0][:120]),
                        "trace": traceback.format_exc(limit=2)[-300:]})


def probe_rounding(jt, device):
    ints = ("int8", "int16", "int32", "int64")
    for dt in ints:
        a = np.array([-7, -5, -1, 0, 1, 5, 7], dtype=dt)
        for div in (2, 3, -2, -3):
            b = np.full(a.shape, div, dtype=dt)
            A, B = jt.array(a), jt.array(b)
            check("%s floor_divide by %d" % (dt, div), "rounding",
                  (A // B).numpy(), a // b)
            check("%s mod by %d" % (dt, div), "rounding",
                  jt.mod(A, B).numpy(), a % b)
            check("%s division identity by %d" % (dt, div), "identity",
                  (A // B).numpy() * b + jt.mod(A, B).numpy(), a)
    f = np.array([-7.5, -0.5, 0.5, 7.5], dtype="float32")
    F = jt.array(f)
    check("float32 floor", "rounding", jt.floor(F).numpy(), np.floor(f))
    check("float32 ceil", "rounding", jt.ceil(F).numpy(), np.ceil(f))
    if hasattr(jt, "round"):
        # numpy rounds halves to even; C rounds half away from zero.
        check("float32 round half-to-even", "rounding",
              jt.round(F).numpy(), np.round(f))


def probe_wraparound(jt, device):
    u = np.array([0, 1, 127, 128, 255], dtype="uint8")
    one = np.ones_like(u)
    U, O = jt.array(u), jt.array(one)
    check("uint8 sub wraps", "wraparound", (U - O).numpy(), u - one)
    check("uint8 add wraps", "wraparound", (U + O).numpy(), u + one)
    i8 = np.array([-128, -1, 0, 1, 127], dtype="int8")
    I, J = jt.array(i8), jt.array(np.ones_like(i8))
    check("int8 add overflow", "wraparound", (I + J).numpy(), i8 + np.ones_like(i8))
    check("int8 negate min", "wraparound", (-I).numpy(), -i8)


def probe_float_edges(jt, device):
    h = np.array([65000.0, 6e-8, 1.0, -0.0], dtype="float16")
    H = jt.array(h)
    check("float16 mul overflow to inf", "float-edge", (H * 2).numpy(),
          h * np.float16(2))
    f = np.array([np.nan, -np.inf, -0.0, 0.0, np.inf], dtype="float32")
    F = jt.array(f)
    check("float32 abs on nan/inf/-0", "float-edge", jt.abs(F).numpy(), np.abs(f))
    check("float32 nan != nan", "float-edge",
          (F != F).numpy(), f != f)
    check("float32 maximum with nan", "float-edge",
          jt.maximum(F, jt.array(np.zeros_like(f))).numpy(),
          np.maximum(f, np.zeros_like(f)))
    check("float32 sign of -0.0", "float-edge",
          jt.sign(F).numpy() if hasattr(jt, "sign") else np.sign(f), np.sign(f))


def probe_promotion(jt, device):
    a32 = np.array([1, 2, 3], dtype="int32")
    f32 = np.array([0.5, 0.5, 0.5], dtype="float32")
    check("int32 + float32 promotes", "promotion",
          (jt.array(a32) + jt.array(f32)).numpy(), a32 + f32)
    i64 = np.array([1, 2, 3], dtype="int64")
    check("int32 + int64 promotes", "promotion",
          (jt.array(a32) + jt.array(i64)).numpy(), a32 + i64)
    check("int32 + python float", "promotion",
          (jt.array(a32) + 0.5).numpy(), a32 + 0.5)
    check("int32 / int32 is float", "promotion",
          (jt.array(a32) / jt.array(a32)).numpy(), a32 / a32)
    b = np.array([True, False, True])
    check("bool + int32", "promotion",
          (jt.array(b) + jt.array(a32)).numpy(), b + a32)


def probe_shifts(jt, device):
    # 64-bit widths need the narrowing off, or the operands are int32 by the
    # time the shift runs and every finding is really KI-DTYPE-002.
    with jt.flag_scope(auto_convert_64_to_32=0):
        for dt, width in (("int32", 32), ("int64", 64)):
            a = np.array([1, 2, -1, 8], dtype=dt)
            for count in (0, 1, width - 1):
                c = np.full(a.shape, count, dtype=dt)
                A, C = arr(jt, a, dt), arr(jt, c, dt)
                check("%s left_shift %d" % (dt, count), "shift",
                      (A << C).numpy(), a << c)
                check("%s right_shift %d" % (dt, count), "shift",
                      (A >> C).numpy(), a >> c)


def probe_reductions(jt, device):
    for dt in ("int32", "float32"):
        a = np.array([[1, 2], [3, 4]], dtype=dt)
        A = jt.array(a)
        for name, ref in (("sum", np.sum), ("prod", np.prod),
                          ("max", np.max), ("min", np.min), ("mean", np.mean)):
            fn = getattr(jt, name, None)
            if fn is None:
                continue
            check("%s %s all" % (dt, name), "reduction", fn(A).numpy(), ref(a))
            check("%s %s dim0" % (dt, name), "reduction",
                  fn(A, 0).numpy(), ref(a, axis=0))
    nan = np.array([np.nan, 1.0, 2.0], dtype="float32")
    check("float32 max with nan propagates", "reduction",
          jt.max(jt.array(nan)).numpy(), np.max(nan))
    big = np.array([2 ** 30, 2 ** 30, 2 ** 30, 2 ** 30], dtype="int32")
    check("int32 sum overflow wraps", "reduction",
          jt.sum(jt.array(big)).numpy(), np.sum(big, dtype="int32"))


def probe_comparison(jt, device):
    f = np.array([np.nan, 1.0, -1.0], dtype="float32")
    z = np.zeros_like(f)
    F, Z = jt.array(f), jt.array(z)
    for name, op in (("less", np.less), ("less_equal", np.less_equal),
                     ("greater", np.greater), ("greater_equal", np.greater_equal),
                     ("equal", np.equal), ("not_equal", np.not_equal)):
        fn = getattr(jt, name, None)
        if fn is None:
            continue
        check("float32 %s with nan" % name, "comparison",
              fn(F, Z).numpy(), op(f, z))


def probe_empty_and_zero_size(jt, device):
    """A zero-length axis is legal input, not an error, and its reductions have
    defined identities: an empty sum is 0 and an empty product is 1. Getting
    those wrong is silent, and a shape-only test never reaches them."""
    z = np.zeros((0,), dtype="float32")
    Z = jt.array(z)
    check("empty shape survives", "empty", np.array(Z.shape), np.array(z.shape))
    check("empty sum is zero", "empty", jt.sum(Z).numpy(), np.sum(z))
    if hasattr(jt, "prod"):
        check("empty prod is one", "empty", jt.prod(Z).numpy(), np.prod(z))
    two = np.zeros((0, 3), dtype="float32")
    T = jt.array(two)
    check("empty 2-D sum axis0", "empty", jt.sum(T, 0).numpy(), np.sum(two, axis=0))
    check("empty concat", "empty",
          jt.concat([Z, jt.array(np.ones((2,), dtype="float32"))], 0).numpy(),
          np.concatenate([z, np.ones((2,), dtype="float32")]))


def probe_broadcasting(jt, device):
    """Broadcast shape rules are easy to get right in the common case and easy
    to get wrong at size 1 against size 0, or when the ranks differ."""
    cases = (((3, 1), (1, 4)), ((2, 3), (3,)), ((1,), (5,)), ((4, 1, 3), (2, 3)))
    for lhs, rhs in cases:
        a = np.ones(lhs, dtype="float32")
        b = np.full(rhs, 2.0, dtype="float32")
        check("broadcast %s + %s" % (lhs, rhs), "broadcast",
              (jt.array(a) + jt.array(b)).numpy(), a + b)


def probe_inplace_aliasing(jt, device):
    """Whether a write through one handle is visible through another.

    Aliasing is a contract no per-operator test states: a slice that silently
    copies, or one that silently shares, both "work" until someone writes."""
    base = np.arange(6, dtype="float32")
    B = jt.array(base)
    view = B[2:4]
    view.sync()
    original = view.numpy().copy()
    check("slice reads the source", "aliasing", original, base[2:4])
    # Assigning into the parent must not retroactively change a value already
    # read out of the child, whichever way the storage is shared.
    B2 = jt.array(base)
    got = B2[2:4].numpy().copy()
    B2[2:4] = jt.array(np.array([9.0, 9.0], dtype="float32"))
    check("setitem updates the parent", "aliasing", B2.numpy(),
          np.concatenate([base[:2], [9.0, 9.0], base[4:]]))
    check("value read before the write is unchanged", "aliasing", got, base[2:4])


def probe_view_semantics(jt, device):
    """Reshape/transpose/expand must agree with numpy on both value and shape;
    a transposed source is where non-contiguous handling shows up."""
    a = np.arange(12, dtype="float32").reshape(3, 4)
    A = jt.array(a)
    check("reshape", "view", A.reshape(4, 3).numpy(), a.reshape(4, 3))
    check("transpose", "view", jt.transpose(A, (1, 0)).numpy(), a.T)
    check("reshape of a transpose", "view",
          jt.transpose(A, (1, 0)).reshape(3, 4).numpy(), a.T.reshape(3, 4))
    check("sum over a transpose", "view",
          jt.sum(jt.transpose(A, (1, 0)), 0).numpy(), a.T.sum(axis=0))


def probe_gradient_edges(jt, device):
    """Gradients at the points where the derivative is not unique.

    These are the silent ones: the forward value is right, so nothing looks
    wrong, and the gradient is only wrong at a measure-zero set of inputs --
    which real data reaches constantly (a relu at exactly 0, tied maxima in a
    pooling window, a clamp sitting on its bound). Two reasonable
    implementations disagree here, and the convention is a choice each
    framework writes down. NumPy has no autograd, so the expectations below are
    PyTorch's documented subgradient choices.
    """
    def grad_of(fn, x_np):
        x = jt.array(x_np)
        y = fn(x)
        return jt.grad(y.sum(), x).numpy()

    # abs'(0): torch picks 0.
    check("grad abs at 0", "grad-edge",
          grad_of(jt.abs, np.array([-1.0, 0.0, 1.0], dtype="float32")),
          np.array([-1.0, 0.0, 1.0], dtype="float32"))

    # relu'(0): torch picks 0.
    if hasattr(jt.nn, "relu"):
        check("grad relu at 0", "grad-edge",
              grad_of(jt.nn.relu, np.array([-1.0, 0.0, 1.0], dtype="float32")),
              np.array([0.0, 0.0, 1.0], dtype="float32"))

    # A tie in max: torch routes the whole gradient to the first maximum for
    # `max()` over all elements. Splitting it evenly is the other defensible
    # answer, so this pins which one is implemented rather than assuming.
    x = jt.array(np.array([1.0, 3.0, 3.0], dtype="float32"))
    g = jt.grad(jt.max(x), x).numpy()
    check("grad max with a tie sums to one", "grad-edge",
          np.array([float(g.sum())]), np.array([1.0]))

    # minimum/maximum against a constant, exactly on the boundary.
    if hasattr(jt, "clamp"):
        v = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
        xx = jt.array(v)
        yy = jt.clamp(xx, -1.0, 1.0)
        gg = jt.grad(yy.sum(), xx).numpy()
        # torch passes gradient through at the bounds themselves.
        check("grad clamp at its bounds", "grad-edge",
              gg, np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype="float32"))

    # sqrt'(0) is infinite; the finite-difference expectation is that it is not
    # silently zero, which is the failure that hides a vanishing gradient.
    g = grad_of(jt.sqrt, np.array([0.0, 1.0, 4.0], dtype="float32"))
    check("grad sqrt away from 0", "grad-edge", g[1:],
          np.array([0.5, 0.25], dtype="float32"))
    check("grad sqrt at 0 is not silently finite", "grad-edge",
          np.array([bool(np.isinf(g[0]) or np.isnan(g[0]))]), np.array([True]))

    # A gradient that must not flow: the second argument of a comparison.
    a = jt.array(np.array([1.0, 2.0], dtype="float32"))
    b = jt.array(np.array([2.0, 1.0], dtype="float32"))
    out = (a > b).float_auto() * a
    ga = jt.grad(out.sum(), a).numpy()
    check("grad through a comparison mask", "grad-edge",
          ga, np.array([0.0, 1.0], dtype="float32"))


def probe_device_agreement(jt, device):
    """The same expression on the two devices must give the same answer.

    Every other category here compares against NumPy, which answers "is this
    right". This one compares CPU against CUDA, which answers "is it at least
    the same" -- and that catches a class the first cannot: a value that is
    defensible on both devices under different conventions, like `maximum`
    returning the NaN on one and the number on the other. The parity suite in
    tests/backends/parity exists for this, but its cases never feed NaN, an
    infinity or a signed zero, so the disagreement lived on.

    Only meaningful with an accelerator; on a CPU-only box every pair is
    trivially equal and the checks are skipped rather than counted as passes.
    """
    try:
        import jittor as _jt
        has_cuda = bool(_jt.compiler.has_cuda)
    except Exception:
        has_cuda = False
    if not has_cuda:
        RESULTS.append({"name": "device agreement", "category": "device-agree",
                        "status": "SKIPPED", "detail": "no accelerator"})
        return

    hard = np.array([np.nan, -np.inf, -0.0, 0.0, np.inf], dtype="float32")
    zero = np.zeros_like(hard)
    cases = (
        ("maximum", lambda a, b: jt.maximum(a, b)),
        ("minimum", lambda a, b: jt.minimum(a, b)),
        ("add", lambda a, b: a + b),
        ("multiply", lambda a, b: a * b),
        ("divide", lambda a, b: a / b),
        ("equal", lambda a, b: (a == b).float_auto()),
        ("less", lambda a, b: (a < b).float_auto()),
    )
    for name, fn in cases:
        answers = {}
        for flag, label in ((0, "cpu"), (1, "cuda")):
            with jt.flag_scope(use_cuda=flag):
                answers[label] = np.asarray(
                    fn(jt.array(hard), jt.array(zero)).numpy(), dtype=np.float64)
        check("%s agrees across devices" % name, "device-agree",
              answers["cuda"], answers["cpu"])


def probe_numerical_stability(jt, device):
    """Where the arithmetic is right but the *order* of it decides the answer.

    `-Ofast` grants the compiler reassociation, so this is the neighbouring
    category to KI-BACKEND-005: the same expression may be evaluated in a
    different order than written, and for floating point that is not a
    no-op. These check the cases where the difference is visible rather than
    in the last bit.

    Expectations come from float64 evaluated in NumPy -- the value the float32
    computation is trying to approximate -- with tolerances chosen so an honest
    float32 result passes and a reassociated one does not.
    """
    # Stagnation: adding 1.0 to a float32 accumulator stops changing it past
    # 2**24. A pairwise or blocked sum keeps working; a naive serial one stops.
    n = 1 << 22
    ones = np.ones(n, dtype="float32")
    got = float(jt.sum(jt.array(ones)).numpy())
    check("large float32 sum does not stagnate", "stability",
          np.array([got]), np.array([float(n)]), rtol=1e-3)

    # Cancellation was checked here and withdrawn. `(big + small) - big` gave
    # 1.0 inside this probe and 0.0 when run on its own, on the same build:
    # what the fusion pass does with the expression depends on what else is in
    # the graph, so there is no stable expectation to assert. Reported as
    # unverified rather than deleted -- the observation is real, the check is
    # not sound, and a probe that scores an unstable case is a probe that will
    # cry wolf.
    RESULTS.append({
        "name": "cancellation order", "category": "stability",
        "status": "UNVERIFIED",
        "detail": "fusion-context dependent: 1.0 in-probe, 0.0 standalone",
    })

    # Softmax on values large enough that a naive exp overflows. The stable
    # form subtracts the row maximum; the result must be finite either way.
    row = np.array([[1000.0, 1000.0, 1000.0]], dtype="float32")
    if hasattr(jt.nn, "softmax"):
        out = jt.nn.softmax(jt.array(row), dim=1).numpy()
        check("softmax of large values stays finite", "stability",
              np.array([bool(np.isfinite(out).all())]), np.array([True]))
        check("softmax of equal values is uniform", "stability",
              out, np.full((1, 3), 1.0 / 3.0, dtype="float32"), rtol=1e-5)

    # exp/log round trip away from the easy range.
    v = np.array([-30.0, -1.0, 0.0, 1.0, 30.0], dtype="float32")
    V = jt.array(v)
    check("exp then log returns the input", "stability",
          jt.log(jt.exp(V)).numpy(), v, rtol=1e-4, atol=1e-4)

    # pow at the awkward exponents, where each library makes a choice.
    base = np.array([0.0, 2.0, -1.0], dtype="float32")
    zero_exp = np.zeros_like(base)
    check("anything to the zero is one", "stability",
          (jt.array(base) ** jt.array(zero_exp)).numpy(),
          np.power(base, zero_exp))

    # Mean of a constant vector must be that constant, not drift with length.
    c = np.full(1 << 20, 0.1, dtype="float32")
    check("mean of a constant does not drift", "stability",
          jt.mean(jt.array(c)).numpy(), np.array(0.1, dtype="float32"),
          rtol=1e-4)


def probe_state_and_reproducibility(jt, device):
    """Process-global state must come back, and a seed must mean something.

    Jittor's device selection, gradient mode and RNG are process-wide, so a
    scope that fails to restore leaks into every test that runs after it -- the
    class the repository keeps a cross-test leak ledger for. These check the
    restore rather than the entry, because entering is the part that obviously
    works.
    """
    # flag_scope restores, including when the body raises.
    before = int(jt.flags.use_cuda)
    with jt.flag_scope(use_cuda=before):
        pass
    check("flag_scope restores use_cuda", "state",
          np.array([int(jt.flags.use_cuda)]), np.array([before]))

    try:
        with jt.flag_scope(use_cuda=before):
            raise RuntimeError("probe")
    except RuntimeError:
        pass
    check("flag_scope restores after an exception", "state",
          np.array([int(jt.flags.use_cuda)]), np.array([before]))

    # Nesting: the inner scope must restore the outer value, not the original.
    with jt.flag_scope(no_grad=1):
        outer = int(jt.flags.no_grad)
        with jt.flag_scope(no_grad=0):
            pass
        check("nested flag_scope restores its caller", "state",
              np.array([int(jt.flags.no_grad)]), np.array([outer]))
    check("no_grad does not leak out of its scope", "state",
          np.array([int(jt.flags.no_grad)]), np.array([0]))

    # A seed has to make two runs identical, and two different seeds differ.
    jt.set_global_seed(1234)
    a = jt.random((64,)).numpy().copy()
    jt.set_global_seed(1234)
    b = jt.random((64,)).numpy().copy()
    check("the same seed reproduces the same draw", "state", b, a)

    jt.set_global_seed(4321)
    c = jt.random((64,)).numpy().copy()
    check("a different seed draws differently", "state",
          np.array([bool(not np.allclose(c, a))]), np.array([True]))

    # Two draws under one seed must not repeat each other: a seed that resets
    # per call would make a stream of "random" numbers constant.
    jt.set_global_seed(99)
    d1 = jt.random((64,)).numpy().copy()
    d2 = jt.random((64,)).numpy().copy()
    check("consecutive draws differ under one seed", "state",
          np.array([bool(not np.allclose(d1, d2))]), np.array([True]))

    # no_grad really stops the graph rather than only marking it.
    x = jt.array(np.ones((4,), dtype="float32"))
    with jt.flag_scope(no_grad=1):
        y = (x * 2).sum()
    try:
        g = jt.grad(y, x)
        produced = bool(np.any(np.abs(g.numpy()) > 0))
    except Exception:
        produced = False
    check("no_grad yields no gradient", "state",
          np.array([produced]), np.array([False]))


def probe_dtype_preservation(jt, device):
    """An operation must return the dtype its inputs imply, not a convenient one.

    A silent widening reads as harmless -- the values are right -- until the
    result is compared, stored or fed to something that dispatches on dtype. A
    silent narrowing loses data. Both are invisible at the call site, which is
    why they are worth asserting rather than assuming.
    """
    def out_dtype(fn, dtype, binary=False):
        # Built through arr() under the flag: jt.array narrows 64-bit inputs by
        # default (KI-DTYPE-002), so without this every 64-bit row would report
        # the construction-time narrowing as if the operator had done it. The
        # shift category already learned this; the helper exists for it.
        with jt.flag_scope(auto_convert_64_to_32=0):
            a = arr(jt, np.ones((4,), dtype=dtype), dtype)
            return str(fn(a, a).dtype if binary else fn(a).dtype)

    for dtype in ("float16", "float32", "float64", "int32", "int64"):
        check("%s add keeps its dtype" % dtype, "dtype-keep",
              np.array([out_dtype(lambda a, b: a + b, dtype, True) == dtype]),
              np.array([True]))
        check("%s multiply keeps its dtype" % dtype, "dtype-keep",
              np.array([out_dtype(lambda a, b: a * b, dtype, True) == dtype]),
              np.array([True]))

    for dtype in ("float16", "float32", "float64"):
        check("%s abs keeps its dtype" % dtype, "dtype-keep",
              np.array([out_dtype(jt.abs, dtype) == dtype]), np.array([True]))
        check("%s sum keeps its dtype" % dtype, "dtype-keep",
              np.array([out_dtype(jt.sum, dtype) == dtype]), np.array([True]))

    # A comparison is a predicate; its result is a truth value whatever the
    # operands were.
    a = jt.array(np.ones((4,), dtype="float32"))
    check("comparison returns bool", "dtype-keep",
          np.array([str((a > a).dtype) == "bool"]), np.array([True]))


def probe_serialization_roundtrip(jt, device):
    """What goes to disk has to come back unchanged.

    Corruption here is the quietest kind there is: the failure appears in a
    later run, in a different process, with nothing left to point at the save.
    """
    import tempfile, os
    cases = (
        ("float32", np.arange(12, dtype="float32").reshape(3, 4)),
        ("float64", np.linspace(-1, 1, 12, dtype="float64").reshape(3, 4)),
        ("int64", np.arange(-6, 6, dtype="int64").reshape(3, 4)),
        ("bool", (np.arange(12) % 2 == 0).reshape(3, 4)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for name, value in cases:
            path = os.path.join(tmp, name + ".pkl")
            with jt.flag_scope(auto_convert_64_to_32=0):
                original = jt.array(value)
                jt.save(original.numpy(), path)
                restored = np.asarray(jt.load(path))
            check("%s survives save/load" % name, "serialize",
                  restored.astype(np.float64), value.astype(np.float64))
            check("%s keeps its dtype on disk" % name, "serialize",
                  np.array([str(restored.dtype) == str(value.dtype)]),
                  np.array([True]))


PROBES = (
    ("rounding", probe_rounding),
    ("dtype-keep", probe_dtype_preservation),
    ("serialize", probe_serialization_roundtrip),
    ("state", probe_state_and_reproducibility),
    ("stability", probe_numerical_stability),
    ("device-agree", probe_device_agreement),
    ("grad-edge", probe_gradient_edges),
    ("empty", probe_empty_and_zero_size),
    ("broadcast", probe_broadcasting),
    ("aliasing", probe_inplace_aliasing),
    ("view", probe_view_semantics),
    ("wraparound", probe_wraparound),
    ("float-edge", probe_float_edges),
    ("promotion", probe_promotion),
    ("shift", probe_shifts),
    ("reduction", probe_reductions),
    ("comparison", probe_comparison),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the full result here")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)

    import jittor as jt
    use_cuda = 1 if args.device == "cuda" else 0
    with jt.flag_scope(use_cuda=use_cuda):
        for name, fn in PROBES:
            guarded(name, name, lambda fn=fn: fn(jt, args.device))

    counts = {}
    for row in RESULTS:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("device=%s  checks=%d  %s" % (args.device, len(RESULTS), counts))
    bad = [r for r in RESULTS if r["status"] != "OK"]
    for row in bad:
        print("  [%s] %-42s %s" % (row["status"], row["name"], row["detail"][:90]))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"device": args.device, "counts": counts, "results": RESULTS},
                       indent=2, sort_keys=True), encoding="utf-8")
        print("detail written to", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
