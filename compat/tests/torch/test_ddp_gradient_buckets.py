"""Captured CUDA buckets retain shared gradients and a rank-agreed order."""
import numpy as np
import pytest
import torch
import torch.distributed as dist
import jittor as jt


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_initialized() or dist.get_world_size() != 2,
    reason='requires two real CUDA ranks',
)]


@jt.flag_scope(backend_fallback='error')
def test_shared_parameter_bucket_order_and_captured_updates():
    rank = dist.get_rank()
    rng = np.random.default_rng(193)
    weights = [rng.normal(0, .15, (8, 8)).astype('float32') for _ in range(3)]
    inputs = [rng.normal(size=(4, 8)).astype('float32') for _ in range(2)]

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.middle = torch.nn.Parameter(torch.tensor(weights[1], device='cuda'))
            self.last = torch.nn.Parameter(torch.tensor(weights[2], device='cuda'))
            # Last registration would put this late, shared gradient first
            # in reverse registration order.
            self.shared = torch.nn.Parameter(torch.tensor(weights[0], device='cuda'))

        def forward(self, x):
            return torch.tanh(x @ self.shared @ self.middle) @ self.last @ self.shared

    module = Shared()
    ddp = torch.nn.parallel.DistributedDataParallel(module, bucket_cap_mb=.0001)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=.03, momentum=.9)

    def step(x):
        loss = ddp(x).square().mean()
        loss.backward()
        gradients = tuple(p.grad.clone() for p in module.parameters())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach(), gradients

    runner = torch.compile(step, mode='reduce-overhead', dynamic=False)
    expected = [w.astype('float64') for w in weights]
    velocity = [np.zeros_like(w) for w in expected]
    x = torch.tensor(inputs[rank], device='cuda')
    before = jt.core.backend_fallback_count()
    try:
        for index in range(10):
            total = [np.zeros_like(w) for w in expected]
            losses = []
            shared, middle, last = expected
            for values in inputs:
                first = values @ shared
                hidden = np.tanh(first @ middle)
                final = hidden @ last
                output = final @ shared
                losses.append(np.mean(output**2))
                dout = 2*output/output.size
                total[0] += final.T @ dout / 2
                dfinal = dout @ shared.T
                total[2] += hidden.T @ dfinal / 2
                dhidden = dfinal @ last.T * (1-hidden**2)
                total[1] += first.T @ dhidden / 2
                total[0] += values.T @ (dhidden @ middle.T) / 2
            loss, gradients = runner(x)
            assert loss.item() == pytest.approx(losses[rank], rel=2e-5, abs=1e-7)
            for gradient, reference in zip(gradients, (total[1], total[2], total[0])):
                np.testing.assert_allclose(gradient.cpu().numpy(), reference, rtol=2e-5, atol=2e-7)
            for i in range(3):
                velocity[i] = .9*velocity[i]+total[i]
                expected[i] = expected[i]-.03*velocity[i]
            for actual, reference in zip(module.parameters(), (expected[1], expected[2], expected[0])):
                np.testing.assert_allclose(actual.detach().cpu().numpy(), reference, rtol=2e-5, atol=2e-7)
        order = ddp._jittor_ddp_state.bucket_param_order
        assert sorted(order) == [0, 1, 2]
        assert order[-1] == 2, order
        assert runner.stats['graph'] >= 4, runner._graph_refused
        assert jt.core.backend_fallback_count() == before
    finally:
        runner.invalidate()
