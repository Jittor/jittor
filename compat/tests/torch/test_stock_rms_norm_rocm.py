"""Standard HF HIP RMSNorm and the portable stock module preserve mutable weights."""

import numpy as np
import pytest
import jittor as jt
import torch
from jittor._runtime import forbid_backend_fallbacks
from jittor._runtime.dispatch import override_kernel, select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


class FixtureRMSNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-1.5, 2, width, device='cuda'))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)).to(dtype)


@pytest.mark.parametrize('kind', ['stock', 'standard'])
def test_standard_provider_and_stock_path_preserve_mutable_weights(kind):
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0, auto_graph_replay=0), forbid_backend_fallbacks(), torch.no_grad():
        raw = np.random.RandomState(91).normal(size=(2, 3, 128)).astype('float32')
        x = torch.tensor(raw, device='cuda')
        if kind == 'stock':
            module = torch.nn.RMSNorm(128, eps=1e-6)
        else:
            module = FixtureRMSNorm(128)
        module.to('cuda')
        actual = select_kernel('nn.rms_norm.inference', x, module.weight, 1e-6)
        assert actual is not None
        calls = []
        def tracked(*args):
            calls.append(True)
            return actual(*args)
        with override_kernel('nn.rms_norm.inference', 'rocm', tracked):
            for scale in [1.0, -2.0]:
                module.weight.fill_(scale)
                y = module(x)
                y.sync()
                assert y.location() == 'device' and y.dtype == torch.float32
                ref = raw.astype('float64')
                ref = ref / np.sqrt((ref**2).mean(-1, keepdims=True) + 1e-6) * scale
                np.testing.assert_allclose(y.cpu().numpy(), ref, rtol=2e-5, atol=2e-5)
        # Stock RMSNorm retains its existing portable path. The standard
        # weighted HF contract is the inference specialization added here.
        assert len(calls) == (2 if kind == "standard" else 0)
    assert jt.core.backend_fallback_count() == before


def test_explicit_forward_override_retains_ownership():
    with jt.runtime.scope(use_rocm=1), forbid_backend_fallbacks(), torch.no_grad():
        x = torch.ones((2, 128), device='cuda')
        module = FixtureRMSNorm(128)
        module.forward = lambda value: value * 3
        np.testing.assert_array_equal(module(x).cpu().numpy(), np.full((2, 128), 3.0))


def test_softmax_frontend_reaches_native_hip_provider():
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, auto_graph_replay=0), forbid_backend_fallbacks(), torch.no_grad():
        raw = np.random.RandomState(109).normal(size=(2, 3, 65)).astype('float32')
        x = torch.tensor(raw, device='cuda')
        kernel = select_kernel('nn.softmax', x, dim=-1)
        assert kernel is not None
        calls = []
        def tracked(*args, **kwargs):
            calls.append(True)
            return kernel(*args, **kwargs)
        with override_kernel('nn.softmax', 'rocm', tracked):
            y = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
        y.sync()
        assert y.location() == 'device' and y.dtype == torch.float32
        expected = raw.astype('float64')
        expected = np.exp(expected - expected.max(-1, keepdims=True))
        expected /= expected.sum(-1, keepdims=True)
        np.testing.assert_allclose(y.cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        assert len(calls) == 1
    assert jt.core.backend_fallback_count() == before
