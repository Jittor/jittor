# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core test utilities, modeled on ``torch.testing._internal.common_utils``.

Provides the pieces every test in the modern suite shares:

  * :func:`make_tensor`  -- seeded input generation with per-dtype value ranges
    (the jittor analogue of ``torch.testing.make_tensor``).
  * :class:`JittorTestCase` -- a ``unittest.TestCase`` with a single, tensor- and
    dtype-aware :meth:`~JittorTestCase.assertEqual` and one tolerance policy, so the
    ~12 divergent local ``check_equal`` helpers and their inconsistent tolerances
    collapse into one place.
  * :func:`parametrize` / :func:`instantiate_parametrized_tests` -- non-device test
    parametrization (mirrors torch's; ``common_device_type`` adds the device axis).
  * dtype group helpers (:func:`floating_types`, :func:`all_types`, ...) used by
    ``OpInfo.dtypes`` to declare which dtypes an op supports.

Design note on oracles (why this catches bugs): forward correctness is asserted
against an INDEPENDENT reference (numpy, via ``OpInfo.ref``), and backward via
``gradcheck`` (numerical vs analytical Jacobian). Neither is jittor-vs-jittor, so a
green test means jittor matches the outside world -- the opposite of the legacy
self-consistency checks the audit flagged as low-assurance.
"""
import os
import unittest
import itertools
import numpy as np

import jittor as jt


# --------------------------------------------------------------------- devices
# jittor exposes a single global accelerator behind ``use_cuda``; on an Ascend
# build that same flag drives ACL, so ``has_acl`` is the only honest CUDA-vs-NPU
# discriminator. A jittor build targets exactly one accelerator.
HAS_ACL = bool(getattr(jt.compiler, "has_acl", 0))
HAS_CUDA = bool(jt.has_cuda)


def get_all_device_types():
    """Device labels this build can actually run (``cpu`` always; one accelerator)."""
    devs = ["cpu"]
    if HAS_ACL:
        devs.append("npu")
    elif HAS_CUDA:
        devs.append("cuda")
    only = os.environ.get("JITTOR_TEST_DEVICES")
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        devs = [d for d in devs if d in want]
    return devs


def use_cuda_for(device):
    """The ``use_cuda`` flag value for a device label (cuda and npu both ride it)."""
    return 0 if device == "cpu" else 1


# ---------------------------------------------------------------------- dtypes
# Canonical jittor dtype strings, grouped the way ``torch.testing._internal.common_dtype``
# groups torch dtypes, so OpInfos can declare ``dtypes=floating_types()`` etc.
bool_ = "bool"
uint8 = "uint8"
int8, int16, int32, int64 = "int8", "int16", "int32", "int64"
float16, bfloat16, float32, float64 = "float16", "bfloat16", "float32", "float64"
complex64, complex128 = "complex64", "complex128"


def _t(*xs):
    return tuple(xs)


def floating_types():
    return _t(float32, float64)


def floating_types_and(*extra):
    return floating_types() + _t(*extra)


def floating_and_complex_types():
    return _t(float32, float64, complex64)


def integral_types():
    return _t(uint8, int8, int16, int32, int64)


def all_types():
    return integral_types() + floating_types()


def all_types_and(*extra):
    return all_types() + _t(*extra)


def complex_types():
    return _t(complex64, complex128)


# numpy dtype for a jittor dtype string (for building reference arrays)
def np_dtype(dtype):
    if dtype == bfloat16:           # numpy has no bfloat16; stage through float32
        return np.float32
    return np.dtype(dtype)


def is_floating(dtype):
    return dtype in (float16, bfloat16, float32, float64)


def is_complex(dtype):
    return dtype in (complex64, complex128)


def is_integral(dtype):
    return dtype in (bool_, uint8, int8, int16, int32, int64)


# ----------------------------------------------------------------- tolerances
# One tolerance policy keyed by dtype, replacing the per-file hard-coded 1e-5 /
# 1e-3 / 1e-2 grab-bag the audit found. (atol, rtol) for forward comparison.
_DEFAULT_TOL = {
    float64: (1e-7, 1e-7),
    float32: (1e-5, 1.3e-6),
    float16: (1e-3, 1e-3),
    bfloat16: (1.6e-2, 1.6e-2),
    complex64: (1e-5, 1.3e-6),
    complex128: (1e-7, 1e-7),
}


def default_tolerances(*dtypes):
    """Loosest (atol, rtol) across the given dtypes; integers compare exactly."""
    atol = rtol = 0.0
    for d in dtypes:
        a, r = _DEFAULT_TOL.get(str(d), (0.0, 0.0))
        atol, rtol = max(atol, a), max(rtol, r)
    return atol, rtol


# ------------------------------------------------------------------ seeding
_seed_counter = itertools.count(0x5EED)


def freeze_rng(seed):
    """Set numpy + jittor RNG seeds for a deterministic block."""
    np.random.seed(seed & 0x7FFFFFFF)
    try:
        jt.set_global_seed(seed & 0x7FFFFFFF)
    except Exception:
        try:
            jt.seed(seed & 0x7FFFFFFF)
        except Exception:
            pass


# --------------------------------------------------------------- make_tensor
# Default [low, high) per dtype, matching torch.testing.make_tensor's table so
# inputs avoid pathological ranges (e.g. no negatives into sqrt unless asked).
_DEFAULT_RANGE = {
    bool_: (0, 2), uint8: (0, 10),
    int8: (-9, 10), int16: (-9, 10), int32: (-9, 10), int64: (-9, 10),
    float16: (-5, 5), bfloat16: (-5, 5), float32: (-9, 9), float64: (-9, 9),
    complex64: (-9, 9), complex128: (-9, 9),
}


def make_tensor(*shape, dtype=float32, low=None, high=None, requires_grad=False,
                noncontiguous=False, exclude_zero=False, seed=None):
    """Create a jittor Var of ``shape``/``dtype`` filled from ``[low, high)``.

    The jittor analogue of ``torch.testing.make_tensor``: deterministic (seeded),
    dtype-appropriate value ranges, optional non-contiguity and zero-exclusion.
    ``requires_grad`` is advisory (jittor leaf Vars are differentiable via
    ``jt.grad``); it tags which inputs a generic test should differentiate.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, jt.NanoVector)):
        shape = tuple(shape[0])
    shape = tuple(int(s) for s in shape)
    rng = np.random.RandomState((seed if seed is not None else next(_seed_counter)) & 0x7FFFFFFF)

    dlo, dhi = _DEFAULT_RANGE.get(dtype, (-9, 9))
    lo = dlo if low is None else low
    hi = dhi if high is None else high

    npd = np_dtype(dtype)
    if dtype == bool_:
        a = rng.randint(0, 2, size=shape).astype(np.bool_)
    elif is_integral(dtype):
        a = rng.randint(int(lo), int(hi), size=shape).astype(npd)
    elif is_complex(dtype):
        re = rng.uniform(lo, hi, size=shape)
        im = rng.uniform(lo, hi, size=shape)
        a = (re + 1j * im).astype(npd)
    else:
        a = rng.uniform(lo, hi, size=shape).astype(npd)

    if exclude_zero:
        if is_floating(dtype) or is_complex(dtype):
            tiny = np.finfo(np.float32).tiny
            a = np.where(np.abs(a) < tiny, npd.type(tiny) if hasattr(npd, "type") else tiny, a)
        else:
            a = np.where(a == 0, 1, a)

    if noncontiguous and a.size > 1:
        # materialize a non-contiguous source then take a strided view, as torch does
        a = np.repeat(a.reshape(-1), 2)[::2].reshape(shape)

    # jt.array silently narrows float64 -> float32, so pin the intended dtype.
    a = np.ascontiguousarray(a)
    v = jt.array(a, dtype=str(a.dtype))
    if dtype == bfloat16:
        v = v.float16()  # closest jittor low-precision; bf16 path validated separately
    elif str(v.dtype) != dtype:
        v = v.cast(dtype)
    if requires_grad:
        try:
            v.requires_grad = True
        except Exception:
            pass
    return v


# ----------------------------------------------------------------- comparison

def to_numpy(x):
    """Materialize a jittor Var / numpy array / python scalar as a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, jt.Var):
        return x.numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def net_scaled_max_err(got, ref):
    """``max|got-ref| / (max|ref|+tiny)`` -- error normalized by the reference scale.

    The metric that does not blow up for an output coordinate whose own magnitude
    is many orders below the dominant one (pure float32 round-off there reads as a
    huge *relative* error while being numerically perfect).
    """
    g = to_numpy(got).astype(np.float64)
    r = to_numpy(ref).astype(np.float64)
    return float(np.max(np.abs(g - r))) / (float(np.max(np.abs(r))) + 1e-30)


class JittorTestCase(unittest.TestCase):
    """``unittest.TestCase`` with a single tensor/dtype-aware equality check.

    :meth:`assertEqual` is the one comparison primitive for the whole suite: it
    accepts jittor Vars, numpy arrays, and python scalars, checks shape first
    (jittor has no 0-d scalar, so a ``(1,)`` vs ``()`` slip is reported as a shape
    error rather than silently broadcast away), then compares with dtype-derived
    tolerances unless overridden.
    """

    # let sub-suites bump precision globally if needed
    precision_override = None

    def assertEqual(self, x, y, atol=None, rtol=None, *, equal_nan=True,
                    exact_dtype=False, msg=None):
        # strings (e.g. dtype names) compare by direct equality, not numerically --
        # to_numpy(str) is a StrDType array and assert_allclose chokes on it.
        if isinstance(x, str) or isinstance(y, str):
            self.assertEqual_(x, y, f"{x!r} != {y!r}; {msg or ''}")
            return
        # scalars / containers
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            self.assertEqual(len(x), len(y), msg=f"len mismatch; {msg or ''}")
            for i, (a, b) in enumerate(zip(x, y)):
                self.assertEqual(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan,
                                 exact_dtype=exact_dtype, msg=f"[{i}] {msg or ''}")
            return

        gx, gy = to_numpy(x), to_numpy(y)
        self.assertEqual_shape(gx, gy, msg)

        if exact_dtype:
            self.assertEqual(str(gx.dtype), str(gy.dtype),
                             msg=f"dtype {gx.dtype} != {gy.dtype}; {msg or ''}")

        if atol is None or rtol is None:
            datol, drtol = default_tolerances(str(gx.dtype), str(gy.dtype))
            if self.precision_override is not None:
                datol = max(datol, self.precision_override)
            atol = datol if atol is None else atol
            rtol = drtol if rtol is None else rtol

        if gx.dtype == np.bool_ or gy.dtype == np.bool_ or \
                (gx.dtype.kind in "iu" and gy.dtype.kind in "iu"):
            np.testing.assert_array_equal(gx, gy, err_msg=msg or "")
        else:
            np.testing.assert_allclose(gx, gy, atol=atol, rtol=rtol,
                                       equal_nan=equal_nan, err_msg=msg or "")

    def assertEqual_shape(self, gx, gy, msg=None):
        self.assertEqual_(tuple(np.shape(gx)), tuple(np.shape(gy)),
                          f"shape {np.shape(gx)} != {np.shape(gy)}; {msg or ''}")

    # raw (non-tensor) equality without recursion into assertEqual; array-safe so a
    # scalar-vs-array mismatch raises a clean assertion, not a numpy truth-value error.
    def assertEqual_(self, a, b, m):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                raise self.failureException(m)
        elif a != b:
            raise self.failureException(m)


# --------------------------------------------------------- parametrize (no device)
# A trimmed, faithful version of torch's parametrize/_TestParametrizer: attaches a
# spec to a test method; instantiate_parametrized_tests expands it into one method
# per param set, suffixing the name. (common_device_type adds the device axis.)

class _ParametrizeSpec:
    def __init__(self, arg_names, values, name_fn=None):
        self.arg_names = arg_names
        self.values = list(values)
        self.name_fn = name_fn


class parametrize:
    """``@parametrize("x", [1, 2])`` or ``@parametrize("a,b", [(1,2),(3,4)])``."""

    def __init__(self, arg_string, values, name_fn=None):
        self.arg_names = [a.strip() for a in arg_string.split(",")]
        self.values = values
        self.name_fn = name_fn

    def __call__(self, fn):
        specs = getattr(fn, "_parametrize_specs", [])
        specs = specs + [_ParametrizeSpec(self.arg_names, self.values, self.name_fn)]
        fn._parametrize_specs = specs
        return fn


def _format_value(v):
    if isinstance(v, (int, str, bool)):
        return str(v)
    if isinstance(v, float):
        return ("%g" % v).replace(".", "_").replace("-", "neg")
    return type(v).__name__


def _expand_parametrize(fn):
    """Yield ``(suffix, kwargs)`` for a possibly multiply-parametrized test fn."""
    specs = getattr(fn, "_parametrize_specs", None)
    if not specs:
        yield "", {}
        return
    combos = [[("", {})]]
    for spec in specs:
        layer = []
        for vals in spec.values:
            vals = vals if isinstance(vals, tuple) and len(spec.arg_names) > 1 else (vals,)
            kwargs = dict(zip(spec.arg_names, vals))
            if spec.name_fn:
                suffix = spec.name_fn(*vals)
            else:
                suffix = "_".join(_format_value(v) for v in vals)
            layer.append((suffix, kwargs))
        combos.append(layer)
    for chosen in itertools.product(*combos):
        suffix = "_".join(s for s, _ in chosen if s)
        kwargs = {}
        for _, k in chosen:
            kwargs.update(k)
        yield suffix, kwargs


def instantiate_parametrized_tests(cls):
    """Expand every ``@parametrize``-decorated method of ``cls`` in place."""
    for name in list(cls.__dict__.keys()):
        fn = cls.__dict__[name]
        if not callable(fn) or not getattr(fn, "_parametrize_specs", None):
            continue
        delattr(cls, name)
        for suffix, kwargs in _expand_parametrize(fn):
            new_name = f"{name}_{suffix}" if suffix else name

            def make(fn=fn, kwargs=kwargs):
                def test(self):
                    return fn(self, **kwargs)
                return test
            method = make()
            method.__name__ = new_name
            method.__doc__ = fn.__doc__
            setattr(cls, new_name, method)
    return cls


# convenience decorators shared with common_device_type
def skipIfNoCUDA(fn):
    return unittest.skipUnless(HAS_CUDA and not HAS_ACL, "requires CUDA build")(fn)


def skipIfNoNPU(fn):
    return unittest.skipUnless(HAS_ACL, "requires Ascend/ACL build")(fn)


def run_tests(argv=None):
    unittest.main(argv=argv, verbosity=2)
