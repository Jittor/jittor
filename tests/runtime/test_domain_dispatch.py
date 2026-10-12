"""Native domain dispatch follows the runtime target and keeps kernel identities."""

from _helpers import capability as _test_capability

import importlib
import math
from types import SimpleNamespace

import jittor as jt
import numpy as np
import pytest

from jittor._runtime.dispatch import dispatch_context, select_kernel


def test_gamma_legacy_accelerators_keep_typed_kernel_registration(monkeypatch):
    gamma = importlib.import_module("jittor.math_util.gamma")
    dispatch = importlib.import_module("jittor._runtime.dispatch")
    # The accelerators keep the device kernels for the widths those kernels
    # were written for; every other width falls through to the primitive-op
    # composite. `lgamma`/`digamma` became type-generic in 37256905 -- their
    # kernels are templated on the element type and `::lgamma` has a double
    # overload, so a float64 Var calls the device's own double-precision
    # routine instead of the ~1e-7 Lanczos series -- while `polygamma`'s CUDA
    # kernel is still hard-coded to `float`, which is what leaves one
    # composite in the float64 row. A half Var has no device kernel on any
    # width's registration, so all three composite.
    rows = {
        "float32": (gamma._gamma_cuda, gamma._digamma_cuda, gamma._polygamma_cuda),
        "float64": (gamma._gamma_cuda, gamma._digamma_cuda, gamma._polygamma_composite),
        "float16": (gamma._lgamma_composite, gamma._digamma_composite,
                    gamma._polygamma_composite),
    }
    for backend in ("cuda", "rocm_legacy", "corex_legacy"):
        for dtype, expected in rows.items():
            context = dispatch.DispatchContext(backend, 0, (dtype,))
            monkeypatch.setattr(dispatch, "dispatch_context", lambda *args, **kwargs: context)
            for operation, kernel in zip(("lgamma", "digamma", "polygamma"), expected):
                assert dispatch.select_kernel("math." + operation) is kernel, (backend, dtype, operation)


def test_legacy_accelerators_share_misc_raw_kernels_and_fft_mode_guard(monkeypatch):
    tensor_ops = importlib.import_module("jittor.misc.tensor_ops")
    legacy = importlib.import_module("jittor.nn.legacy_complex")
    dispatch = importlib.import_module("jittor._runtime.dispatch")
    implementations = {
        "misc.repeat_interleave_dim0": tensor_ops._repeat_interleave_dim0_cuda.__wrapped__,
        "misc.stack_no_grad": tensor_ops._stack_no_grad_cuda_fast.__wrapped__,
        "misc.unbind_no_grad": tensor_ops._unbind_no_grad_cuda_fast.__wrapped__,
        "misc.unique_code": tensor_ops._unique_code_cuda,
        "misc.scan_2d": tensor_ops._scan_2d_cuda,
        "nn.legacy_fft2": legacy._fft2_cuda,
    }
    for backend in ("cuda", "rocm_legacy", "corex_legacy"):
        for operation, implementation in implementations.items():
            assert dispatch.registered_kernel(operation, backend) is implementation
        context = dispatch.DispatchContext(backend, 0, ("float32",))
        monkeypatch.setattr(dispatch, "dispatch_context", lambda *args, **kwargs: context)
        runtime = SimpleNamespace(use_cuda=1)
        monkeypatch.setattr(jt, "runtime", runtime)
        assert dispatch.select_kernel("nn.legacy_fft2", None) is legacy._fft2_cuda
        runtime.use_cuda = 2
        assert dispatch.select_kernel("nn.legacy_fft2", None) is None


def test_cpu_domain_kernel_choices_and_values():
    gamma = importlib.import_module("jittor.math_util.gamma")
    tensor_ops = importlib.import_module("jittor.misc.tensor_ops")
    with jt.flag_scope(use_cuda=0):
        x = jt.array([0.5, 1.5, 3.0], dtype="float32")
        assert dispatch_context(x).backend == "cpu"
        assert select_kernel("math.lgamma", x) is gamma._gamma_cpu
        assert select_kernel("misc.scan_2d", x, False) is tensor_ops._scan_2d_cpu
        assert select_kernel("misc.stack_no_grad", (x, x), 0) is None
        np.testing.assert_allclose(
            gamma.lgamma.apply(x).numpy(), [math.lgamma(v) for v in (0.5, 1.5, 3.0)],
            rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(jt.cumsum(x, 0).numpy(), [0.5, 2.0, 5.0])


def test_fft_cache_uses_tensor_device_and_reuses_matrices():
    fft = importlib.import_module("jittor.fft")
    with jt.flag_scope(use_cuda=0):
        x = jt.array([1.0, 2.0, 3.0, 4.0])
        first = fft._dft_mats(4, False, x)
        second = fft._dft_mats(4, False, x)
        assert first is second
        context = dispatch_context(x)
        assert (4, False, context.backend, context.device_id) in fft._dft_mat_cache
        actual = fft.fft(x)
        np.testing.assert_allclose(actual.real.numpy(), np.fft.fft([1, 2, 3, 4]).real,
                                   atol=1e-5)
        np.testing.assert_allclose(actual.imag.numpy(), np.fft.fft([1, 2, 3, 4]).imag,
                                   atol=1e-5)


@pytest.mark.skipif(not _test_capability.check_accelerator('cuda', backend=jt).enabled or not jt.compiler.is_cuda, reason="requires CUDA")
def test_cpu_tensor_can_feed_the_cuda_runtime_target():
    gamma = importlib.import_module("jittor.math_util.gamma")
    tensor_ops = importlib.import_module("jittor.misc.tensor_ops")
    with jt.flag_scope(use_cuda=0):
        x = jt.array([1.0, 2.0, 3.0])
        x.sync()
        assert x.location() == "cpu"
    with jt.flag_scope(use_cuda=1):
        assert dispatch_context(x).backend == "cuda"
        assert select_kernel("math.lgamma", x) is gamma._gamma_cuda
        assert select_kernel("misc.scan_2d", x, False) is tensor_ops._scan_2d_cuda
        result = jt.cumsum(x, 0)
        result.sync()
        assert result.location() == "device"
        np.testing.assert_allclose(result.numpy(), [1.0, 3.0, 6.0])


@pytest.mark.skipif(not _test_capability.check_accelerator('cuda', backend=jt).enabled or not jt.compiler.is_cuda, reason="requires CUDA")
def test_cuda_gamma_kernels_follow_the_element_type():
    gamma = importlib.import_module("jittor.math_util.gamma")
    with jt.flag_scope(use_cuda=1):
        single = jt.array([0.5, 1.5, 3.0], dtype="float32")
        double = single.float64()
        assert dispatch_context(single).backend == "cuda"
        assert select_kernel("math.lgamma", single) is gamma._gamma_cuda
        # lgamma/digamma follow the Var's width (37256905); polygamma's CUDA
        # kernel is still a `float*` one, so a double Var keeps the composite.
        assert select_kernel("math.lgamma", double) is gamma._gamma_cuda
        assert select_kernel("math.digamma", double) is gamma._digamma_cuda
        assert select_kernel("math.polygamma", double) is gamma._polygamma_composite
        np.testing.assert_allclose(
            gamma.lgamma.apply(single).numpy(), [math.lgamma(v) for v in (0.5, 1.5, 3.0)],
            rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            gamma.lgamma.apply(double).numpy(), [math.lgamma(v) for v in (0.5, 1.5, 3.0)],
            rtol=1e-5, atol=1e-5)
