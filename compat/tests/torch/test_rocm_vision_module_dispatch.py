"""Native module construction preserves the registered ROCm provider contract."""

import math
from unittest import mock

import numpy as np
import pytest
import jittor as jt
import torch
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import registered_kernel, override_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


@pytest.fixture(autouse=True)
def eager_inference():
    with jt.flag_scope(auto_graph_replay=0):
        yield


def fixture(kind, strided=False):
    if kind == 'conv':
        layer = torch.nn.Conv2d(4, 6, 3, padding=1).cuda()
        shape = (2, 4, 5, 7)
    elif kind == 'group':
        layer = torch.nn.GroupNorm(2, 4).cuda()
        shape = (2, 4, 5, 7)
    else:
        layer = torch.nn.LayerNorm(7).cuda()
        shape = (2, 4, 5, 7)
    raw = np.random.RandomState(131).normal(size=shape).astype('float32')
    x = torch.tensor(raw.transpose(0, 1, 3, 2).copy(), device='cuda').transpose(-1, -2) if strided else torch.tensor(raw, device='cuda')
    return layer, torch.nn.Sequential(layer), x


def functional(kind, layer, x):
    if kind == 'conv':
        return jt.nn.conv2d(x, layer.weight, layer.bias, layer.stride,
                            layer.padding, layer.dilation, layer.groups)
    if kind == 'group':
        return jt.nn.group_norm(x, layer.num_groups, layer.weight, layer.bias, layer.eps)
    return jt.nn.layer_norm(x, layer.normalized_shape, layer.weight, layer.bias, layer.eps)


def op(kind):
    return {'conv': 'conv2d', 'group': 'nn.group_norm', 'layer': 'nn.layer_norm.inference'}[kind]


@pytest.mark.parametrize('kind', ['conv', 'group', 'layer'])
@pytest.mark.parametrize('strided', [False, True])
def test_native_module_builds_same_kernel_without_python_provider(kind, strided):
    from jittor.backends.rocm.kernels import convolution, normalization
    owner = convolution if kind == 'conv' else normalization
    name = {'conv': 'conv2d', 'group': 'group_norm', 'layer': 'layer_norm'}[kind]
    with torch.no_grad():
        layer, outer, x = fixture(kind, strided)
        expected = functional(kind, layer, x).cpu().numpy().copy()
        outer(x).sync()
        with mock.patch.object(owner, name, side_effect=AssertionError('Python provider reached')):
            actual = outer(x)
            actual.sync()
        assert isinstance(actual, torch.Tensor) and actual.device.type == 'cuda'
        assert actual.location() == 'device' and actual.dtype == torch.float32
        assert not actual.requires_grad
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['conv', 'group', 'layer'])
def test_registered_override_and_predicate_error_are_not_bypassed(kind):
    with torch.no_grad():
        layer, outer, x = fixture(kind)
        expected = outer(x).cpu().numpy().copy()
        original = registered_kernel(op(kind), 'rocm')
        calls = []
        def traced(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        with override_kernel(op(kind), 'rocm', traced):
            actual = outer(x)
            actual.sync()
        assert len(calls) == 1
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-5, atol=3e-6)
        checks = []
        def fail(*args, **kwargs):
            checks.append(True)
            raise RuntimeError('vision provider predicate failed')
        with override_kernel(op(kind), 'rocm', traced, supports=fail):
            with pytest.raises(RuntimeError, match='vision provider predicate failed'):
                outer(x)
        assert len(checks) == 1
        np.testing.assert_allclose(outer(x).cpu().numpy(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['conv', 'group', 'layer'])
def test_public_function_replacement_and_module_hooks_keep_ownership(kind):
    name = {'conv': 'conv2d', 'group': 'group_norm', 'layer': 'layer_norm'}[kind]
    with torch.no_grad():
        layer, outer, x = fixture(kind)
        expected = outer(x).cpu().numpy().copy()
        replacement = torch.full(tuple(expected.shape), 3.0, device='cuda')
        with mock.patch.object(jt.nn, name, return_value=replacement) as changed:
            np.testing.assert_array_equal(outer(x).cpu().numpy(), np.full(expected.shape, 3.0))
            assert changed.call_count == 1
        handle = layer.register_forward_hook(lambda module, args, output: output + 2)
        try:
            np.testing.assert_allclose(outer(x).cpu().numpy(), expected + 2, rtol=2e-5, atol=3e-6)
        finally:
            handle.remove()
        layer.forward = lambda value: replacement
        np.testing.assert_array_equal(outer(x).cpu().numpy(), np.full(expected.shape, 3.0))


def test_conv_forward_and_attribute_descriptors_are_resolved_normally():
    class CustomConv(torch.nn.Conv2d):
        def _conv_forward(self, value, weight, bias):
            return value * 5
    class CustomNorm(torch.nn.LayerNorm):
        @property
        def eps(self):
            return self._epsilon * 100
        @eps.setter
        def eps(self, value):
            self._epsilon = value
    with torch.no_grad():
        _, _, x = fixture('conv')
        layer = CustomConv(4, 6, 3, padding=1).cuda()
        np.testing.assert_allclose(torch.nn.Sequential(layer)(x).cpu().numpy(), x.cpu().numpy()*5)
        norm = CustomNorm(7).cuda()
        expected = jt.nn.layer_norm(x, (7,), norm.weight, norm.bias, norm.eps)
        np.testing.assert_allclose(torch.nn.Sequential(norm)(x).cpu().numpy(), expected.cpu().numpy(), rtol=2e-5, atol=3e-6)


def test_silu_offer_reuses_native_source_and_mutable_parameters():
    from jittor.backends.rocm.kernels import normalization
    with torch.no_grad():
        layer, _, x = fixture('group', True)
        outer = torch.nn.Sequential(layer, torch.nn.SiLU())
        outer(x).sync()
        for scale in [1.0, -1.5]:
            layer.weight.fill_(scale)
            layer.bias.fill_(.25)
            expected = jt.nn.silu(functional('group', layer, x)).cpu().numpy().copy()
            with mock.patch.object(normalization, '_normalize', side_effect=AssertionError('Python norm builder reached')):
                actual = outer(x)
                actual.sync()
            assert actual.location() == 'device'
            np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-5, atol=3e-6)


def test_custom_attribute_lookup_keeps_its_epsilon():
    class DynamicNorm(torch.nn.LayerNorm):
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            return value * 100 if name == 'eps' else value
    with torch.no_grad():
        _, _, x = fixture('layer')
        layer = DynamicNorm(7).cuda()
        expected = functional('layer', layer, x).cpu().numpy().copy()
        outer = torch.nn.Sequential(layer)
        outer(x).sync()
        np.testing.assert_allclose(outer(x).cpu().numpy(), expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['conv', 'group', 'layer'])
def test_dynamic_code_output_uses_actual_shape(kind):
    with torch.no_grad():
        layer, outer, x = fixture(kind)
        outer(x).sync()
        # The declared first dimension is an allocation bound. Execution sets
        # the actual length before the normalization/convolution consumes it.
        dynamic = jt.code((-2, 4, 5, 7), 'float32', [x],
            cuda_header='#include <hip/hip_runtime.h>',
            cuda_src='out0->set_shape({1,4,5,7}); CHECK(hipMemcpyAsync(out0_p,in0_p,4*5*7*sizeof(float),hipMemcpyDeviceToDevice)==hipSuccess);')
        assert tuple(dynamic.shape) == (1, 4, 5, 7)
        actual = outer(dynamic)
        actual.sync()
        assert actual.location() == 'device'
        expected = functional(kind, layer, x[:1])
        np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('kind', ['conv', 'group', 'layer'])
def test_gradient_enabled_call_keeps_all_requested_derivatives(kind):
    layer, outer, x = fixture(kind)
    with torch.no_grad():
        outer(x).sync()
    x = x.detach().clone().requires_grad_(True)
    y = outer(x)
    seed = torch.tensor(np.random.RandomState(132).normal(size=tuple(y.shape)).astype('float32'), device='cuda')
    loss = (y*seed).sum()
    grads = torch.autograd.grad(loss, (x, layer.weight, layer.bias))
    reference = functional(kind, layer, x)
    wanted = torch.autograd.grad((reference*seed).sum(), (x, layer.weight, layer.bias))
    for actual, expected in zip(grads, wanted):
        actual.sync()
        expected.sync()
        assert actual is not None and actual.location() == 'device'
        np.testing.assert_allclose(actual.detach().cpu().numpy(), expected.detach().cpu().numpy(), rtol=3e-5, atol=3e-5)


def test_rocm_native_linear_bias_provider_and_override():
    from jittor.backends.rocm.libraries import install as rocm_libraries
    with torch.no_grad():
        layer = torch.nn.Linear(7, 11).cuda()
        outer = torch.nn.Sequential(layer)
        x = torch.tensor(np.random.RandomState(137).normal(size=(2, 3, 7)).astype("float32"),
                         device="cuda")
        expected = jt.nn.linear(x, layer.weight, layer.bias).cpu().numpy().copy()
        outer(x).sync()
        with mock.patch.object(rocm_libraries, "_hipblas_matmul",
                               side_effect=AssertionError("Python provider reached")):
            for run in (lambda: outer(x), lambda: layer(x),
                        lambda: jt.nn.linear(x, layer.weight, layer.bias)):
                actual = run()
                actual.sync()
                assert actual.location() == "device" and not actual.requires_grad
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=3e-5, atol=3e-6)
        original = registered_kernel("linear_bias_accumulator", "rocm")
        calls = []
        def traced(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        with override_kernel("linear_bias_accumulator", "rocm", traced):
            for run in (lambda: outer(x), lambda: layer(x),
                        lambda: jt.nn.linear(x, layer.weight, layer.bias)):
                actual = run()
                actual.sync()
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=3e-5, atol=3e-6)
        assert len(calls) == 3


def test_rocm_native_linear_respects_bias_descriptor():
    class CustomLinear(torch.nn.Linear):
        pass

    with torch.no_grad():
        layer = CustomLinear(7, 11).cuda()
        outer = torch.nn.Sequential(layer)
        x = torch.tensor(np.random.RandomState(139).normal(size=(2, 3, 7)).astype("float32"),
                         device="cuda")
        expected = outer(x).cpu().numpy().copy()
        with mock.patch.object(CustomLinear, "bias",
                               property(lambda self: self.__dict__["bias"] + 2), create=True):
            actual = outer(x)
            actual.sync()
            assert actual.location() == "device"
            np.testing.assert_allclose(actual.cpu().numpy(), expected + 2, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rocm_exact_gelu_inference_on_device(dtype):
    raw = np.random.RandomState(138).uniform(-6., 6., (4, 197, 23)).astype("float32")
    with torch.no_grad():
        x = torch.tensor(raw, device="cuda", dtype=dtype)
        values = x.float().cpu().numpy()
        expected = values * 0.5 * (1. + np.vectorize(math.erf)(values / math.sqrt(2.)))
        y = torch.nn.functional.gelu(x)
        y.sync()
        assert y.location() == "device" and not y.requires_grad and y.dtype == dtype
        tolerance = 3e-5 if dtype == torch.float32 else 0.008
        np.testing.assert_allclose(y.float().cpu().numpy(), expected, rtol=tolerance, atol=2e-6)


def test_rocm_int_mm_uses_int32_and_preserves_out():
    rng = np.random.RandomState(180)
    a = rng.randint(-17, 18, size=(19, 32)).astype("int8")
    b = rng.randint(-17, 18, size=(32, 23)).astype("int8")
    x = torch.tensor(a, device="cuda")
    y = torch.tensor(b.T.copy(), device="cuda").t()
    expected = a.astype("int32") @ b.astype("int32")
    z = torch._int_mm(x, y)
    z.sync()
    assert z.location() == "device" and z.dtype == torch.int32
    np.testing.assert_array_equal(z.cpu().numpy(), expected)
    out = torch.empty((19, 23), dtype=torch.int32, device="cuda")
    assert torch._int_mm(x, y, out=out) is out
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
    with pytest.raises(RuntimeError, match="int8"):
        torch._int_mm(x.float(), y)


def test_rocm_tf32_policy_does_not_leak_to_strict_matmul():
    rng = np.random.RandomState(181)
    a = rng.normal(size=(128, 128)).astype("float32")
    b = rng.normal(size=(128, 128)).astype("float32")
    x, y = torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda")
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        strict = (x @ y).cpu().numpy().copy()
        torch.backends.cuda.matmul.allow_tf32 = True
        relaxed = (x @ y).cpu().numpy().copy()
        torch.backends.cuda.matmul.allow_tf32 = False
        strict_again = (x @ y).cpu().numpy().copy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    np.testing.assert_array_equal(strict, strict_again)
    ref = a.astype("float64") @ b.astype("float64")
    np.testing.assert_allclose(strict, ref, rtol=3e-5, atol=2e-5)
    # Hardware that cannot provide XF32 can legally retain full precision.
    assert np.linalg.norm(relaxed-ref)/np.linalg.norm(ref) < 0.005


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_autocast_matmul_linear_and_batched(dtype):
    x = torch.ones(2, 8, 64, device="cuda")
    w = torch.ones(64, 17, device="cuda")
    batched = torch.ones(2, 64, 17, device="cuda")
    layer = torch.nn.Linear(64, 17).cuda().eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for value in (x[0] @ w, x @ w, x @ batched, layer(x)):
            value.sync()
            assert value.location() == "device" and value.dtype == dtype
        np.testing.assert_array_equal((x @ w).float().cpu().numpy(), np.full((2, 8, 17), 64))
