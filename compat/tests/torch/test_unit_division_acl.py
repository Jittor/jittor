"""Exact unit division retains values, independent storage and gradients.

310P's FP16 RealDiv saturates NaN/Inf to finite values. Unit division uses its
mathematical identity instead; the expected special values here come from the
independent CPU arithmetic contract, not the vendor's saturation behavior.
"""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(params=("cpu", "acl"))
def target(request):
    acl = request.param == "acl"
    if acl and not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(acl), use_acl=int(acl),
                          transpose_storage_view=1), forbid_backend_fallbacks():
        yield "cuda" if acl else "cpu"
    assert jt.core.backend_fallback_count() == before


def _special(dtype):
    value_type = np.dtype(dtype).type
    limits = np.finfo(dtype)
    return np.array([0., -0., 1., -1., .1, -.5, np.nan, np.inf, -np.inf,
                     limits.max, -limits.max, limits.tiny, -limits.tiny,
                     np.nextafter(value_type(0), value_type(1)),
                     np.nextafter(value_type(0), value_type(-1)), 3.], dtype=dtype)


def _host(value):
    return value.detach().cpu().numpy()


def _same(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    zeros = np.isfinite(expected) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("shape", ((1, 16), (0, 16), ()))
def test_unit_division_special_values_and_empty_scalar_shapes(target, dtype, shape):
    values = (_special(dtype).reshape(shape) if shape == (1, 16)
              else np.empty(shape, dtype=dtype) if shape == (0, 16)
              else np.asarray(-0., dtype=dtype))
    with torch.no_grad():
        source = torch.tensor(values, dtype=getattr(torch, dtype), device=target)
        output = source / 1.0
        jt.sync_all(True)
        assert output.dtype == source.dtype and tuple(output.shape) == shape
        assert output.location() == ("device" if target == "cuda" else "cpu")
        if output.numel():
            assert output.data_ptr() != source.data_ptr()
        _same(_host(output), values)
        _same(_host(source), values)


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_unit_division_storage_and_weighted_gradient(target, dtype):
    values = np.array([[1., -2.], [3., -4.]], dtype=dtype)
    tensor_dtype = getattr(torch, dtype)
    with torch.no_grad():
        source = torch.tensor(values, dtype=tensor_dtype, device=target)
        output = source / 1.0
        jt.sync_all(True)
        source.add_(7.)
        jt.sync_all(True)
        _same(_host(output), values)
        output.add_(11.)
        jt.sync_all(True)
        _same(_host(source), (values + 7).astype(dtype))
    source = torch.tensor(values, dtype=tensor_dtype, device=target, requires_grad=True)
    weights = torch.tensor([[.125, -.25], [.375, -.5]], dtype=tensor_dtype, device=target)
    output = source / 1.0
    gradient, = torch.autograd.grad((output * weights).sum(), (source,))
    jt.sync_all(True)
    assert output.requires_grad and gradient.dtype == source.dtype
    assert gradient.location() == ("device" if target == "cuda" else "cpu")
    _same(_host(gradient), np.array([[.125, -.25], [.375, -.5]], dtype=dtype))


@pytest.mark.parametrize("kind", ("unit", "reflected", "integer", "boolean", "numpy",
                                  "tensor", "other_float", "strided", "integer_input", "explicit_cpu"))
def test_unit_copy_guard_leaves_other_division_contracts_on_their_path(target, kind, monkeypatch):
    from jittor.compat.torch.installers.tensor import method_api
    device = "cpu" if kind == "explicit_cpu" else target
    source = torch.tensor([[2., 4.], [8., 16.]], device=device,
                          dtype=torch.int32 if kind == "integer_input" else torch.float32)
    opname, divisor = "__truediv__", 1.0
    expected = np.array([[2., 4.], [8., 16.]], dtype=np.float32)
    if kind == "reflected":
        opname = "__rtruediv__"; expected = 1. / expected
    elif kind == "integer": divisor = 1
    elif kind == "boolean": divisor = True
    elif kind == "numpy": divisor = np.float64(1.)
    elif kind == "tensor": divisor = torch.tensor(1., device=device)
    elif kind == "other_float": divisor = 2.; expected /= 2.
    elif kind == "strided":
        source = source.transpose(0, 1); expected = expected.T
        assert not source.is_contiguous()
    calls = []
    original = method_api._binary_native

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(method_api, "_binary_native", observed)
    output = method_api._true_division(source, divisor, opname)
    jt.sync_all(True)
    # This checks the performance branch's boundary too: numerically equivalent
    # CPU inputs and other divisor spellings must keep their original dispatch.
    selected = target == "cuda" and kind == "unit"
    assert bool(calls) != selected
    _same(_host(output), np.asarray(expected, dtype=np.float32))
