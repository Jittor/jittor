# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Device parity: every op's forward AND backward must agree across CPU and CUDA/NPU.

This closes the single most important gap in the op battery: ``gradcheck`` runs on
CPU only, so it verifies the derivative *formula* (device-independent) but NOT that
the accelerator *kernel* computes it correctly. Yet this project's most dangerous
bugs are exactly device-specific silent-wrong kernels — CUDA scatter min/max
dropping contributions (``880cd6ad``), int8/int16 reduce miscompiles (``eb3c8bee``),
setitem negative-index backward corruption (``58e95b73``). Those are invisible to a
CPU-only check.

For every op in ``op_db`` this runs the forward and backward on **identical inputs**
on CPU and on the accelerator and asserts they match (float32 tolerance). CPU is the
oracle (serial, well-tested); a divergence means the accelerator kernel is wrong —
which is precisely how the scatter bug was originally caught. Skipped entirely on a
CPU-only build (reported, not silent).

Run::  python -m pytest tests/backends/parity/test_device_parity.py
"""
import gc
import sys
import traceback
import unittest

import numpy as np
import jittor as jt

from _helpers.common import (
    JittorTestCase, to_numpy, net_scaled_max_err, per_element_max_rel_err,
    HAS_CUDA, HAS_ACL,
)
from _helpers import reference_cache
from opinfo.database import op_db

_ACCEL = "npu" if HAS_ACL else ("cuda" if HAS_CUDA else None)
_PARITY_DTYPES = ("float32", "int8", "int16")

def _maybe_np(x):
    return to_numpy(x) if isinstance(x, jt.Var) else x


def _is_float_np(a):
    return isinstance(a, np.ndarray) and a.dtype.kind == "f"


def _normalize_outputs(out):
    """A list of Var outputs: unwrap a values/indices namedtuple to its values,
    flatten a multi-output (slogdet, split) tuple to its Var elements."""
    if hasattr(out, "values") and not isinstance(out, jt.Var):
        return [out.values]
    if isinstance(out, (list, tuple)):
        return [o for o in out if isinstance(o, jt.Var)]
    return [out]


# Ops whose accelerator path has a known issue. Compile failures cannot be run as
# expected failures in this aggregate process: Jittor's asynchronous executor keeps
# the failed op and rethrows it in the next test. Their strict xfail coverage lives in
# tests/ops/test_ops.py; parity skips the same unsupported path with an explicit reason.
_LINALG_OPS = ("det", "slogdet", "inv", "solve", "cholesky", "qr", "svd")


def _cuda_linalg_works():
    """Probe whether jittor's CUDA linalg actually runs here, instead of hard-coding a
    skip. The factorizations route through ``jt.numpy_code`` whose CUDA path uses cupy;
    on a CUDA-toolkit/cupy mismatch that nvrtc compile can genuinely fail. A blanket
    ``cupy unavailable`` skip is dishonest when cupy DOES work (it hides ops that pass
    AND would mask a real future CPU/CUDA linalg divergence) -- so probe once and skip
    only on a genuine failure, surfacing the real error as the reason. Probe CuPy
    directly: a failed Jittor async op would remain queued and poison the next test.
    Materializing the result is required because ``jt.numpy_code`` is not executed by
    a plain Var ``sync()`` on this path."""
    if _ACCEL != "cuda":
        return False, f"no CUDA accelerator (_ACCEL={_ACCEL})"
    try:
        import cupy
        from cupy.cuda.compiler import CompileException
    except ImportError as e:  # pragma: no cover - depends on the hardware environment
        return False, f"cupy CUDA linalg unavailable: ImportError: {str(e)[:120]}"
    try:
        # Match the maintained determinant samples. Smaller inputs stay on CuPy's
        # C++11 reduction path and miss the CUB C++17 compile used by batched linalg.
        eye = cupy.eye(3, dtype=cupy.float32)
        for shape in ((2, 3, 3), (2, 2, 3, 3)):
            a = cupy.broadcast_to(eye, shape).copy()
            cupy.asnumpy(cupy.linalg.det(a))
        return True, ""
    except (CompileException, RuntimeError) as e:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "cupy is installed but its CUDA linalg probe failed; refusing to "
            "skip device parity: {}: {}".format(type(e).__name__, str(e)[:120])
        ) from e


_KNOWN_DEVICE_ISSUES = {}
if _ACCEL == "npu":
    _KNOWN_DEVICE_ISSUES.update({
        name: ("skip", "sub-32-bit integer reduction atomics are not verified on NPU")
        for name in ("sum_int_reduce", "prod_int_reduce", "max_int_reduce", "min_int_reduce")
    })
    _KNOWN_DEVICE_ISSUES.update({
        name: ("skip", "logical reduction atomics are not verified on NPU")
        for name in ("all_bool_reduce", "any_bool_reduce")
    })

def _cotangent(shape, index):
    """The fixed random direction output ``index`` is projected onto.

    ``standard_normal(shape)`` rather than ``randn(*shape)``: they are the same
    draw from the same seeded generator for every non-empty shape, but
    ``randn()`` with no arguments -- a **0-d output** -- returns a python float
    instead of an array, and the ``.astype`` that followed raised
    ``AttributeError``. Every operator whose forward is a single number (``sum``,
    ``trace``, and every loss) therefore never reached its backward comparison,
    which is the one thing this battery exists to do. See
    ``tests/backends/parity/test_parity_harness.py``.
    """
    return np.asarray(
        np.random.RandomState(1234 + index).standard_normal(tuple(shape)),
        dtype="float32")


def _run(op, sample, use_cuda):
    """Run op forward+backward on one device from the sample's materialized numpy.

    Returns (forward_np, [grad_np...]) using a fixed seeded cotangent so the two
    devices are compared on the same projection.
    """
    np_in = _maybe_np(sample.input)
    np_args = [_maybe_np(a) for a in sample.args]
    with jt.flag_scope(use_cuda=use_cuda):
        inp = jt.array(np_in, dtype=str(np.asarray(np_in).dtype)) \
            if isinstance(np_in, np.ndarray) else jt.array(np_in)
        args = [jt.array(a, dtype=str(a.dtype)) if isinstance(a, np.ndarray) else a
                for a in np_args]
        out = op.op(inp, *args, **sample.kwargs)
        outs = _normalize_outputs(out)            # handle namedtuple / multi-output
        fwds = [to_numpy(o) for o in outs]
        grads = None
        if op.supports_autograd:
            diff = [v for v in [inp, *args]
                    if isinstance(v, jt.Var) and str(v.dtype) in ("float32", "float64")]
            float_outs = [o for o in outs if str(o.dtype).startswith("float")]
            if diff and float_outs:
                loss = None
                for j, o in enumerate(float_outs):
                    cot = _cotangent(np.asarray(to_numpy(o)).shape, j)
                    # A 0-d output already *is* the scalar the projection
                    # produces, so scale it rather than building a 0-d Var and
                    # reducing that: jittor's codegen cannot reduce a 0-d Var
                    # (``reduce.add`` on ``float32[]`` fails the op compiler's
                    # step limit), and the two are the same number.
                    term = (o * float(cot) if cot.ndim == 0
                            else (o * jt.array(cot)).sum())
                    loss = term if loss is None else loss + term
                grads = [to_numpy(g) for g in jt.grad(loss, diff)]
        return fwds, grads


# The CPU side of every comparison, kept between runs (0.22).
#
# Measured on this battery: the CPU half is 18% of a cold run's wall clock and
# 26% of a warm one, so it is a fraction rather than the "half" the task
# assumed -- the dominant term is kernel compilation, and most of that is on
# the accelerator side. What makes the fraction worth taking is that a cached
# value skips the CPU-side *compiles* too, not just the microseconds of
# execution.
#
# What keeps it honest is in reference_cache: the key carries a content hash of
# ``python/jittor`` and of the materialized inputs, so an entry can only have
# been produced by byte-identical code answering a byte-identical question, and
# every reuse is counted and printed in the terminal summary.
# ``JITTOR_REFERENCE_CACHE=0`` recomputes everything, which is what a release
# gate should set when it wants the oracle re-derived from scratch.
_REFERENCE_CACHE = reference_cache.ReferenceCache("device-parity")

#: Part of the cache key: how ``_run`` draws its cotangent. Changing the
#: projection changes every gradient, so an entry from before such a change has
#: to miss rather than be compared against the new one.
_COTANGENT_SCHEME = "standard_normal(RandomState(1234+j))-v1"


def _oracle_material(op, sample, index, dtype):
    return reference_cache.key_material(
        "device-parity-cpu-oracle", op.full_name, dtype, index,
        _maybe_np(sample.input),
        [_maybe_np(a) for a in sample.args],
        sample.kwargs,
        _COTANGENT_SCHEME,
    )


def _cpu_oracle(op, sample, index, dtype):
    """``_run(..., use_cuda=0)``, from the cache when it holds this exact answer.

    A failure is never cached: an operator that raised on the CPU has no oracle,
    and storing the exception would turn one broken run into a permanent one.
    """
    material = _oracle_material(op, sample, index, dtype)
    stored = _REFERENCE_CACHE.load(material)
    if stored is not None:
        values, extras = stored
        count = int(extras["forward_count"])
        grads = values[count:] if extras["has_grads"] else None
        return values[:count], grads
    fwds, grads = _run(op, sample, use_cuda=0)
    _REFERENCE_CACHE.store(
        material, list(fwds) + list(grads or []),
        {"forward_count": len(fwds), "has_grads": grads is not None,
         "operator": op.full_name, "dtype": dtype, "sample": index})
    return fwds, grads


@unittest.skipUnless(_ACCEL is not None, "device parity needs an accelerator (CUDA/NPU)")
class TestDeviceParity(JittorTestCase):
    """One generated test per op: CPU forward/backward == accelerator forward/backward."""

    _linalg_ok = False
    _linalg_reason = "linalg availability was not probed"

    # float32 kernels on CPU vs accelerator: allow accumulation-order round-off but
    # nothing larger (a real kernel bug is orders of magnitude past this). TWO metrics,
    # both must hold -- they fail to different bug classes:
    #  * net-scaled (global): max|err| / max|ref|. Catches gross / near-peak divergence.
    #  * per-element: max_i |err_i|/(|ref_i|+atol). Catches a wrong coordinate whose own
    #    magnitude is a non-trivial fraction of -- but still far BELOW -- the dominant one
    #    (e.g. a dropped 0.1%-of-peak gradient: net-scaled reads 1e-3 and PASSES at
    #    GRAD_TOL, per-element reads ~1 and FAILS). This is the scatter/int-reduce/setitem
    #    silent-wrong class the global metric is too coarse for.
    # PE_ATOL is the key knob: it is the magnitude below which a coordinate is "negligible"
    # AND where float32 catastrophic cancellation lives. A near-zero TRUE gradient computed
    # by cancelling O(1) terms carries absolute reorder noise ~eps*(intermediate scale),
    # NOT ~eps*(result) -- measured up to ~1.8e-6 for SDPA backward on a 1e-6-magnitude
    # coord (CPU matches the f64 truth to 1e-9 only by accumulation-order luck; CUDA's
    # order leaves ~1e-6 -- legitimate reorder, not a kernel bug). PE_ATOL=1e-3 absorbs
    # that (floor = GRAD_PE_TOL*PE_ATOL = 5e-6 absolute, ~3x over the worst observed noise)
    # while still catching any drop/corruption of a coordinate above ~1e-3 in magnitude.
    FWD_TOL = 2e-4
    GRAD_TOL = 2e-3
    FWD_PE_TOL = 2e-3      # per-element relative, forward
    GRAD_PE_TOL = 5e-3     # per-element relative, backward
    PE_ATOL = 1e-3         # negligibility/cancellation floor for the per-element metric

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The parallel op compiler stays ON here (0.16). It used to be switched off for
        # this whole class, with the reason "otherwise an asynchronous compile error can
        # surface in the following test and be attributed to the wrong operator". That
        # reason is real but the remedy was the wrong one: it cost this battery its
        # compile parallelism -- the dominant term in the CUDA gate, ~227 operators
        # compiling one kernel at a time -- to work around a *lifetime* bug, not a
        # concurrency one.
        #
        # The mechanism, measured rather than assumed: a Var whose op failed to compile
        # stays in the graph for as long as anything references it, and every later
        # ``sync_all(True)`` re-raises that same failure. What holds the reference after
        # a failed test is the exception's traceback -- the failing frame's locals. So
        # the next operator's test really is blamed for this operator's kernel, and
        # serialising the compiler never addressed that: it only made the first report
        # arrive with a smaller fused op in it. ``_check`` below drops those frames, so
        # the failure stays where it happened whether or not the compiler is parallel.
        # ``tests/compiler/test_parallel_compile_attribution.py`` is the proof, on CPU.
        #
        # Probe only when this accelerator class actually runs. Collection must not
        # compile or synchronize CUDA kernels.
        cls._linalg_ok, cls._linalg_reason = _cuda_linalg_works()

    def _check(self, op, dtype="float32"):
        """One operator's parity check, with its runtime state contained.

        A failure here must not travel: see ``setUpClass``. ``clear_frames`` releases
        the Vars the failing frames hold (pytest keeps the traceback for the report),
        so the poisoned op is collectable before the next test builds a graph. The
        report keeps its message and line numbers; only ``--showlocals`` gets less.
        """
        try:
            self._compare(op, dtype)
        except Exception:
            traceback.clear_frames(sys.exc_info()[2])
            gc.collect()
            raise

    def _compare(self, op, dtype="float32"):
        if op.full_name in _LINALG_OPS and not self._linalg_ok:
            self.skipTest(self._linalg_reason)
        requires_grad = dtype.startswith("float")
        samples = op.sample_inputs("cpu", dtype, requires_grad=requires_grad)
        n = 0
        for i, s in enumerate(samples):
            f_cpu, g_cpu = _cpu_oracle(op, s, i, dtype)
            f_acc, g_acc = _run(op, s, use_cuda=1)
            self.assertEqual(len(f_cpu), len(f_acc),
                             msg=f"{op.full_name} output count cpu vs {_ACCEL} sample#{i}")
            for j, (fc, fa) in enumerate(zip(f_cpu, f_acc)):
                self.assertEqual(tuple(fc.shape), tuple(fa.shape),
                                 msg=f"{op.full_name} fwd[{j}] shape cpu vs {_ACCEL} sample#{i}")
                if not _is_float_np(fc):
                    self.assertTrue(
                        np.array_equal(fa, fc),
                        f"{op.full_name} FORWARD[{j}] differs cpu vs {_ACCEL} "
                        f"[{dtype}] sample#{i}: {fa!r} != {fc!r}",
                    )
                    continue
                input_size = np.asarray(_maybe_np(s.input)).size
                reduce_size = max(1, input_size // max(1, fc.size))
                accumulation_floor = np.sqrt(reduce_size) * np.finfo(fc.dtype).eps
                fwd_tol = max(self.FWD_TOL, accumulation_floor)
                ferr = net_scaled_max_err(fa, fc)
                self.assertLess(ferr, fwd_tol,
                                f"{op.full_name} FORWARD[{j}] differs cpu vs {_ACCEL} "
                                f"(net-scaled {ferr:.2e}) sample#{i} -- accelerator kernel suspect")
                fpe = per_element_max_rel_err(fa, fc, atol=self.PE_ATOL)
                self.assertLess(fpe, self.FWD_PE_TOL,
                                f"{op.full_name} FORWARD[{j}] per-element divergence cpu vs "
                                f"{_ACCEL} (max rel {fpe:.2e}) sample#{i} -- a sub-peak output "
                                f"coordinate is wrong (silent-wrong kernel class)")
            if g_cpu is not None and g_acc is not None:
                for k, (gc, ga) in enumerate(zip(g_cpu, g_acc)):
                    gerr = net_scaled_max_err(ga, gc)
                    self.assertLess(gerr, self.GRAD_TOL,
                                    f"{op.full_name} BACKWARD (input {k}) differs cpu vs "
                                    f"{_ACCEL} (net-scaled {gerr:.2e}) sample#{i} -- "
                                    f"accelerator backward kernel suspect")
                    gpe = per_element_max_rel_err(ga, gc, atol=self.PE_ATOL)
                    self.assertLess(gpe, self.GRAD_PE_TOL,
                                    f"{op.full_name} BACKWARD (input {k}) per-element divergence "
                                    f"cpu vs {_ACCEL} (max rel {gpe:.2e}) sample#{i} -- a sub-peak "
                                    f"gradient coordinate is wrong (scatter/setitem silent-wrong class)")
            n += 1
        self.assertGreater(n, 0, f"{op.full_name}: no samples")


def _install():
    for op in op_db:
        def make(o, dtype):
            def test(self):
                self._check(o, dtype)
            test.__doc__ = \
                f"device parity (cpu vs accelerator): {o.full_name} [{dtype}]"
            return test
        accel = _ACCEL or "cuda"
        for dtype in _PARITY_DTYPES:
            if not (op.supports_dtype(dtype, "cpu") and
                    op.supports_dtype(dtype, accel)):
                continue
            method = make(op, dtype)
            action_reason = _KNOWN_DEVICE_ISSUES.get(op.full_name)
            if action_reason is not None:
                action, reason = action_reason
                if action == "skip":
                    method = unittest.skip(reason)(method)
                elif action == "xfail":
                    method = unittest.expectedFailure(method)
                    method.__doc__ = f"KNOWN-BUG (expected failure): {reason}"
            suffix = "" if dtype == "float32" else "_" + dtype
            setattr(TestDeviceParity, f"test_{op.full_name}{suffix}", method)


_install()


if __name__ == "__main__":
    unittest.main(verbosity=2)
