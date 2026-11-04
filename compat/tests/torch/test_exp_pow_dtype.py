"""Public low-precision exp/pow preserve dtype through downstream contractions.

References are NumPy float64 formulas on the actual rounded inputs, including
both input gradients. The native AMP white-list remains a separate policy.
"""
import math

import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability as _test_capability


_DEVICES = ["cpu"] + (["cuda"] if _test_capability.any_accelerator_enabled(backend=jt) else [])
_SPECS = [(device, dtype) for device in _DEVICES
          for dtype in ("float16", "float32", "bfloat16")
          if device == "cpu" or dtype != "bfloat16" or not jt.compiler.has_acl]
_ACL_HF32_CONTRACTION_TOLERANCE = 1e-3


def _numpy(value):
    return value.detach().float().cpu().numpy().astype(np.float64)


def _close(value, expected, dtype):
    tolerance = {"float16": 0.004, "bfloat16": 0.025, "float32": 2e-5}[dtype]
    np.testing.assert_allclose(_numpy(value), expected, rtol=tolerance, atol=tolerance)


def _close_contraction(value, expected, dtype, device):
    if dtype == 'float32' and device != 'cpu' and jt.compiler.has_acl:
        # 310P's opted-in FP32 contractions use the reduced-precision SDK
        # policy. The independent native Torch NPU probe matches these
        # results exactly. Scalar exp/pow keep their tighter tolerances.
        np.testing.assert_allclose(_numpy(value), expected,
            rtol=_ACL_HF32_CONTRACTION_TOLERANCE, atol=1e-4)
    else:
        _close(value, expected, dtype)


@pytest.fixture(autouse=True)
def _device_scope_and_no_fallbacks(request):
    before = jt.core.backend_fallback_count()
    device = request.node.callspec.params['device']
    # Native scalar factories follow the ambient backend. Exercise each
    # declared device under its own runtime scope, including on NPU hosts.
    with jt.runtime.scope(use_cuda=int(device != 'cpu')):
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("device,dtype", _SPECS)
@pytest.mark.parametrize("spelling", ["method", "function"])
def test_exp_forward_and_input_gradient(device, dtype, spelling):
    x = torch.tensor([-2., -.25, .5, 2.], dtype=getattr(torch, dtype),
                     device=device, requires_grad=True)
    source = _numpy(x)
    output = x.exp() if spelling == "method" else torch.exp(x)
    assert output.dtype == x.dtype
    _close(output, np.exp(source), dtype)
    output.sum().backward()
    assert x.grad.dtype == x.dtype
    _close(x.grad, np.exp(source), dtype)


@pytest.mark.parametrize("device,dtype", _SPECS)
@pytest.mark.parametrize("spelling", ["operator", "method", "function", "scalar_base", "scalar_function"])
def test_pow_forward_and_both_input_gradients(device, dtype, spelling):
    tensor_dtype = getattr(torch, dtype)
    base = torch.tensor([1.25, 2., 3.5], dtype=tensor_dtype, device=device, requires_grad=True)
    exponent = torch.tensor([.25, .5, 1.5], dtype=tensor_dtype, device=device, requires_grad=True)
    a, b = _numpy(base), _numpy(exponent)
    if spelling == "operator":
        output = base ** exponent
    elif spelling == "method":
        output = base.pow(exponent)
    elif spelling == "function":
        output = torch.pow(base, exponent)
    elif spelling == "scalar_base":
        output = 10.0 ** exponent
        a = np.full_like(b, 10.)
    else:
        output = torch.pow(10.0, exponent)
        a = np.full_like(b, 10.)
    assert output.dtype == tensor_dtype
    expected = a ** b
    _close(output, expected, dtype)
    output.sum().backward()
    _close(exponent.grad, expected * np.log(a), dtype)
    if spelling in ("operator", "method", "function"):
        assert base.grad.dtype == tensor_dtype
        _close(base.grad, b * a ** (b - 1), dtype)


@pytest.mark.parametrize("device,dtype", _SPECS)
def test_cosine_attention_and_sincos_embedding_need_no_input_casts(device, dtype):
    tensor_dtype = getattr(torch, dtype)
    x = torch.tensor(np.arange(1, 65, dtype=np.float32).reshape(1, 2, 4, 8) / 32,
                     dtype=tensor_dtype, device=device)
    q = torch.nn.functional.normalize(x, dim=-1)
    scale = torch.tensor(np.full((2, 1, 1), math.log(10), np.float32),
                         dtype=tensor_dtype, device=device)
    scale = torch.clamp(scale, max=math.log(100)).exp()
    probabilities = torch.nn.functional.softmax((q @ q.transpose(-2, -1)) * scale, dim=-1)
    context = probabilities @ x
    assert probabilities.dtype == context.dtype == x.dtype
    reference_x = _numpy(x)
    reference_q = reference_x / np.linalg.norm(reference_x, axis=-1, keepdims=True)
    scores = reference_q @ reference_q.swapaxes(-2, -1) * _numpy(scale)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probability_reference = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    expected_context = probability_reference @ reference_x
    _close_contraction(context, expected_context, dtype, device)

    omega = torch.arange(8, device=device).to(tensor_dtype) / 8
    frequency = 1.0 / (10000.0 ** omega)
    grid = torch.arange(4, device=device).to(tensor_dtype)
    positions = grid[:, None] @ frequency[None]
    assert frequency.dtype == positions.dtype == tensor_dtype
    _close_contraction(positions, np.arange(4)[:, None] / (10000.0 ** _numpy(omega))[None], dtype, device)


@pytest.mark.parametrize("device,dtype", _SPECS)
@pytest.mark.parametrize("p", [1., 2., 3., float("inf")])
def test_normalize_dtype_and_input_gradient(device, dtype, p):
    x = torch.tensor([[.5, 1.25, -2., 4.], [1., 2., -3., 5.]],
                     dtype=getattr(torch, dtype), device=device, requires_grad=True)
    source = _numpy(x)
    absolute = np.abs(source)
    if p == float("inf"):
        norm = absolute.max(axis=-1, keepdims=True)
        norm_derivative = np.sign(source) * (absolute == norm)
    else:
        norm = (absolute ** p).sum(axis=-1, keepdims=True) ** (1 / p)
        norm_derivative = np.sign(source) * absolute ** (p - 1) / norm ** (p - 1)
    output = torch.nn.functional.normalize(x, p=p, dim=-1, eps=.25)
    assert output.dtype == x.dtype
    _close(output, source / norm, dtype)
    output.sum().backward()
    expected_gradient = 1 / norm - source.sum(axis=-1, keepdims=True) * norm_derivative / norm ** 2
    _close(x.grad, expected_gradient, dtype)


@pytest.mark.parametrize("device,dtype", _SPECS)
def test_normalize_clamped_small_norm_uses_typed_epsilon(device, dtype):
    x = torch.tensor([[.01, .02, -.03]], dtype=getattr(torch, dtype),
                     device=device, requires_grad=True)
    source = _numpy(x)
    output = torch.nn.functional.normalize(x, dim=-1, eps=.25)
    assert output.dtype == x.dtype
    _close(output, source / .25, dtype)
    output.sum().backward()
    _close(x.grad, np.full_like(source, 4.), dtype)


@pytest.mark.parametrize("device", _DEVICES)
def test_integer_pow_and_native_white_policy_stay_separate(device):
    value = torch.tensor([2, 3, 4], dtype=torch.int32, device=device)
    for output in (value ** 2, torch.pow(value, 2)):
        assert output.dtype == torch.int32
        np.testing.assert_array_equal(_numpy(output), [4, 9, 16])
    with jt.flag_scope(amp_reg=0):
        native = jt.array([.25, .5], dtype="float16")
        if device == "cuda":
            native = native.cuda()
        assert str(jt.exp(native).dtype) == "float32"
        assert str(jt.pow(native, native).dtype) == "float32"


@pytest.mark.parametrize("device", _DEVICES)
def test_existing_explicit_autocast_policy_and_flags_are_preserved(device):
    x = torch.tensor([.25, .5], dtype=torch.float16, device=device)
    before = jt.flags.amp_reg
    with torch.autocast(device_type=device, dtype=torch.float16):
        active = jt.flags.amp_reg
        # This change only repairs non-autocast public boundaries. Existing
        # native white-op inference inside explicit autocast remains intact.
        assert x.exp().dtype == torch.float32
        assert torch.exp(x).dtype == torch.float32
        assert (10.0 ** x).dtype == torch.float32
        assert torch.pow(x, x).dtype == torch.float32
        assert jt.flags.amp_reg == active
    assert jt.flags.amp_reg == before


@pytest.mark.parametrize("device", _DEVICES)
def test_power_keywords_and_existing_out_rejection(device):
    x = torch.tensor([1., 2., 3.], dtype=torch.float16, device=device)
    for value in (x.pow(exponent=2.), torch.pow(input=x, exponent=2.)):
        assert value.dtype == x.dtype
        np.testing.assert_array_equal(_numpy(value), [1., 4., 9.])
    # Native delegates previously reject out; retain an explicit error until
    # its aliasing, dtype and autograd semantics have independent coverage.
    with pytest.raises(TypeError):
        torch.exp(x, out=torch.empty_like(x))
    with pytest.raises(TypeError):
        torch.pow(x, 2., out=torch.empty_like(x))
