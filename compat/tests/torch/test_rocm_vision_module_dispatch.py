"""Native module construction preserves the registered ROCm provider contract."""

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
