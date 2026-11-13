"""Dense captured addmm preserves both weight layouts and all three gradients."""
import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.cuda
def test_captured_rectangular_addmm_matches_fp64_forward_and_gradients(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('requires actual CUDA')
    from jittor.backends.cuda.kernels.cublas import lt_linear_cuda as lt
    rng = np.random.default_rng(52)
    raw = [rng.normal(0, .1, shape).astype('float32')
           for shape in ((640, 384), (384, 1152), (1152,))]
    dy = rng.normal(0, .01, (640, 1152)).astype('float32')
    calls = []
    original = lt._LtLinearProduct.execute

    def observe(self, *args):
        calls.append((self.transposed_weight, self.fuse_bias))
        return original(self, *args)

    monkeypatch.setattr(lt._LtLinearProduct, 'execute', observe)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source, self.weight, self.bias = [
                torch.nn.Parameter(torch.tensor(value, device='cuda')) for value in raw]

        def forward(self):
            return torch.addmm(self.bias, self.source, self.weight)

    module = Projection()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.)
    delta = torch.tensor(dy, device='cuda')

    def step():
        out = module()
        (out*delta).sum().backward()
        gradients = tuple(p.grad.detach().clone() for p in module.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return out.detach(), gradients

    before = jt.core.backend_fallback_count()
    runner = torch.compile(step, mode='reduce-overhead', dynamic=False)
    try:
        for index in range(8):
            source = raw[0]+index*.001
            with torch.no_grad():
                module.source.copy_(torch.tensor(source, device='cuda'))
            out, gradients = runner()
            x, w, b, g = [v.astype('float64') for v in (source, raw[1], raw[2], dy)]
            expected = (x@w+b, g@w.T, x.T@g, g.sum(0))
            for value, reference in zip((out, *gradients), expected):
                assert value.device.type == 'cuda'
                np.testing.assert_allclose(value.cpu().numpy(), reference, atol=1e-5, rtol=1e-4)
        assert (True, True) in calls
        assert runner.stats['graph'] >= 3, runner._graph_refused
        assert jt.core.backend_fallback_count() == before
    finally:
        runner.invalidate()


@pytest.mark.parametrize('alpha,beta', [(1., 1.), (2., .5), (1., 0.)])
def test_addmm_cpu_scalar_and_matrix_bias_fallbacks(alpha, beta):
    x = np.arange(12, dtype='float64').reshape(3, 4)/7
    w = np.arange(20, dtype='float64').reshape(4, 5)/11
    for shape in ((5,), (3, 5)):
        bias = np.full(shape, .25, dtype='float64')
        out = torch.addmm(torch.tensor(bias, device='cpu'),
                          torch.tensor(x, device='cpu'), torch.tensor(w, device='cpu'),
                          alpha=alpha, beta=beta)
        np.testing.assert_allclose(out.numpy(), alpha*(x@w)+beta*bias, atol=1e-12, rtol=1e-12)
