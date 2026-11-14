"""torch.optim.AdamW takes the fused list update on CUDA by default.

torch's default AdamW (fused=None, foreach=None) is a multi-tensor kernel on
CUDA. The per-parameter update here built about ten graph nodes per tensor,
and on a 450-tensor diffusers UNet that step alone outweighed the forward:
ddpm training went 210.7 -> 105.6 ms a step with the fused update. What the
fused path must preserve is the answer, and the caller's choice to opt out.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

import jittor as jt
from _helpers import capability as _test_capability


def _train(fused=None, steps=4):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 33), torch.nn.GELU(),
                                torch.nn.Linear(33, 4)).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1, fused=fused)
    rng = np.random.RandomState(1)
    for _ in range(steps):
        x = torch.tensor(rng.randn(8, 16).astype("float32"), device="cuda")
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach().cpu().numpy() for p in model.parameters()]


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestAdamwFusedByDefault(unittest.TestCase):
    def setUp(self):
        from jittor.backends.cuda.kernels.optim import fused_adamw_cuda
        self.module = fused_adamw_cuda
        self.calls = []
        original = fused_adamw_cuda._cuda_fused_adamw_updates

        def counted(*args):
            self.calls.append(1)
            return original(*args)

        from jittor._runtime.dispatch import override_kernel
        scope = override_kernel("optim.adamw_fused", "cuda", counted,
                                supports=fused_adamw_cuda._supports)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_the_default_is_fused_and_matches_the_per_parameter_update(self):
        with jt.flag_scope(use_cuda=1):
            reference = _train(fused=False)
            self.assertEqual(self.calls, [])
            fused = _train()
        self.assertEqual(len(self.calls), 4)
        for want, got in zip(reference, fused):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)

    def test_provider_iterator_is_not_consumed_before_writeback(self):
        from jittor._runtime.dispatch import override_kernel
        with jt.flag_scope(use_cuda=1):
            reference = _train(fused=False)
            consumed = []

            def iterator_updates(*args):
                consumed.append(1)
                yield from self.module._cuda_fused_adamw_updates(*args)

            with override_kernel("optim.adamw_fused", "cuda", iterator_updates,
                                 supports=self.module._supports):
                actual = _train()
        self.assertEqual(len(consumed), 4)
        for want, got in zip(reference, actual):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)
    def test_fused_write_keeps_aliases_versions_and_pending_snapshots(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
        alias = p.detach()
        optimizer = torch.optim.AdamW([p], lr=.1)
        state_alias = None
        for _ in range(3):
            before = p.detach().cpu().numpy().copy()
            snapshot = p.detach().clone()
            version = p._autograd_version()
            pointer = p.data_ptr()
            p.grad = torch.ones_like(p)
            optimizer.step()
            assert p.data_ptr() == pointer == alias.data_ptr()
            assert p._autograd_version() > version
            assert alias._autograd_version() == p._autograd_version()
            assert p.is_leaf and p.requires_grad
            np.testing.assert_array_equal(snapshot.cpu().numpy(), before)
            np.testing.assert_array_equal(alias.cpu().numpy(), p.detach().cpu().numpy())
            moment = optimizer.state[p]['exp_avg']
            if state_alias is None:
                state_alias = moment.detach()
            assert state_alias.data_ptr() == moment.data_ptr()
            np.testing.assert_array_equal(state_alias.cpu().numpy(), moment.cpu().numpy())

    def test_batched_and_individual_writeback_match_every_optimizer_state(self):
        def run(batched):
            p = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
            q = torch.nn.Parameter(torch.tensor([3., 4.], device='cuda'))
            opt = torch.optim.AdamW([p, q], lr=.01)
            from jittor.compat.torch import optimizer_adam
            original = optimizer_adam._optimizer_update_many
            decisions = []

            def writer(targets, values):
                result = original(targets, values) if batched else False
                decisions.append(result)
                return result

            with patch.object(optimizer_adam, '_optimizer_update_many', writer):
                for step in range(4):
                    p.grad = torch.full_like(p, step + 1.)
                    q.grad = None if step == 1 else torch.full_like(q, .5)
                    opt.step()
            assert decisions == [batched] * 4
            return [t.detach().cpu().numpy().copy() for param in (p, q)
                    for t in (param, opt.state[param]['exp_avg'], opt.state[param]['exp_avg_sq'],
                              opt.state[param]['step'])]

        for actual, expected in zip(run(True), run(False)):
            np.testing.assert_array_equal(actual, expected)

    def test_fused_batch_respects_a_custom_tensor_writer(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
        optimizer = torch.optim.AdamW([p], lr=.01)
        p.grad = torch.ones_like(p)
        calls = []
        original = type(p)._optimizer_update

        def writer(target, value):
            calls.append(target)
            return original(target, value)

        with patch.object(type(p), '_optimizer_update', writer):
            optimizer.step()
        assert any(target is p for target in calls)

    def test_zero_decay_adam_preserves_dynamic_onecycle_and_live_state(self):
        def run(**options):
            p = torch.nn.Parameter(torch.tensor([1., -2., .3], device='cuda'))
            frozen = torch.nn.Parameter(torch.ones((3,), device='cuda'), requires_grad=False)
            optimizer = torch.optim.Adam([frozen, p], lr=.01, **options)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.03, total_steps=12)
            observations = []
            for index in range(12):
                p.grad = torch.tensor([index/10 + .1, -.2, .5], device='cuda')
                optimizer.step()
                state = optimizer.state[p]
                assert state['step'].device.type == 'cpu'
                observations.extend(value.detach().cpu().numpy().copy()
                                    for value in (p, state['exp_avg'], state['exp_avg_sq'], state['step']))
                scheduler.step()
            np.testing.assert_array_equal(frozen.detach().cpu().numpy(), np.ones(3))
            return observations

        reference = run(fused=False)
        assert not self.calls
        for actual, expected in zip(run(), reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
        assert len(self.calls) == 12
        self.calls.clear()
        run(foreach=False)
        assert not self.calls

    def test_adam_coupled_decay_stays_on_the_coupled_update(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
        optimizer = torch.optim.Adam([p], lr=.1, weight_decay=.2)
        p.grad = torch.zeros_like(p)
        optimizer.step()
        assert not self.calls
        np.testing.assert_allclose(p.detach().cpu().numpy(), [.9, 1.9], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
