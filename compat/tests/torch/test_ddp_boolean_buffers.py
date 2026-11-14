"""Boolean DDP buffers use byte-copy NCCL broadcasts on real CUDA ranks."""
import numpy as np
import pytest
import torch
import torch.distributed as dist
import jittor as jt


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_initialized() or dist.get_world_size() < 2,
    reason='requires a real multi-rank CUDA process group',
)]


def pattern(rank, step):
    return (np.arange(6).reshape(2, 3)+rank+step) % 3 == 0


class BooleanBuffers(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(rank+1), device='cuda'))
        self.register_buffer('mask', torch.tensor(pattern(rank, 0), device='cuda'), persistent=False)
        self.register_buffer('empty', torch.empty((0,), dtype=torch.bool, device='cuda'), persistent=False)
        self.register_buffer('offset', torch.tensor(float(rank), device='cuda'))

    def forward(self, value):
        return value*self.weight*(self.mask.float()+self.offset)


@pytest.mark.parametrize('broadcast_buffers', (True, False))
@jt.flag_scope(backend_fallback='error')
def test_boolean_buffers_preserve_aliases_and_synchronize_before_forward(broadcast_buffers):
    assert hasattr(torch, '_torch_compat_install_context')
    before = jt.core.backend_fallback_count()
    rank, world = dist.get_rank(), dist.get_world_size()
    module = BooleanBuffers(rank)
    alias = module.mask.detach()
    ddp = torch.nn.parallel.DistributedDataParallel(module, broadcast_buffers=broadcast_buffers)
    assert module.weight.item() == 1.
    source = 0 if broadcast_buffers else rank
    np.testing.assert_array_equal(alias.cpu().numpy(), pattern(source, 0))
    assert module.mask.dtype == module.empty.dtype == torch.bool
    assert module.empty.numel() == 0
    for step in range(3):
        with torch.no_grad():
            module.mask.copy_(torch.tensor(pattern(rank, step), device='cuda'))
            module.offset.fill_(rank+step)
        output = ddp(torch.full((2, 3), float(rank+1), device='cuda'))
        assert output.device.type == 'cuda'
        output.sum().backward()
        expected = pattern(source, step)
        np.testing.assert_array_equal(alias.cpu().numpy(), expected)
        np.testing.assert_array_equal(output.detach().cpu().numpy(),
                                      (rank+1)*(expected+source+step))
        gradients = [np.sum((peer+1)*(pattern(0 if broadcast_buffers else peer, step)
                                     +(0 if broadcast_buffers else peer)+step))
                     for peer in range(world)]
        assert module.weight.grad.item() == pytest.approx(float(np.mean(gradients)), abs=1e-6)
        module.zero_grad(set_to_none=True)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize('broadcast_buffers', (True, False))
@pytest.mark.parametrize('grouped', (True, False))
@jt.flag_scope(backend_fallback='error')
def test_captured_step_repeats_buffer_broadcasts_and_preserves_aliases(
        monkeypatch, broadcast_buffers, grouped):
    from jittor.compat import collectives
    if not grouped:
        original = collectives._nccl_ops
        class LegacyCollectives:
            nccl_broadcast_many = None
            def __getattr__(self, name):
                return getattr(original(), name)
        monkeypatch.setattr(collectives, '_nccl_ops', LegacyCollectives)
    rank = dist.get_rank()

    class BufferedStep(BooleanBuffers):
        def __init__(self, rank):
            super().__init__(rank)
            self.register_buffer('extra_offset', torch.zeros((3,), device='cuda'))
            self.register_buffer('extra_mask', torch.tensor(
                (np.arange(5)+rank) % 2 == 0, device='cuda'))

        def forward(self, value, phase):
            previous = self.mask.detach().clone()
            previous_extra = self.extra_mask.detach().clone()
            with torch.no_grad():
                self.offset.add_(rank+1)
                self.extra_offset.add_(2*(rank+1))
                self.mask.copy_((torch.arange(6, device='cuda').reshape(2, 3)
                                 + phase + rank) % 3 == 0)
                self.extra_mask.copy_((torch.arange(5, device='cuda')+phase+rank) % 2 == 0)
            return (value*self.weight+self.offset, previous, self.offset.detach().clone(),
                    previous_extra, self.extra_offset.detach().clone())

    module = BufferedStep(rank)
    with torch.no_grad():
        module.weight.fill_(1.)
        module.offset.zero_()
    offset_alias, mask_alias = module.offset.detach(), module.mask.detach()
    extra_offset_alias, extra_mask_alias = module.extra_offset.detach(), module.extra_mask.detach()
    ddp = torch.nn.parallel.DistributedDataParallel(module, broadcast_buffers=broadcast_buffers)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=.01, momentum=.9)

    def step(value, phase):
        output, previous, offset, previous_extra, extra_offset = ddp(value, phase)
        loss = output.square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss.detach(), previous, offset, previous_extra, extra_offset

    runner = torch.compile(step, mode='reduce-overhead', dynamic=False)
    before = jt.core.backend_fallback_count()
    try:
        for index in range(12):
            loss, previous, offset, previous_extra, extra_offset = runner(
                torch.tensor([rank+1.], device='cuda'),
                torch.tensor([index], dtype=torch.int64, device='cuda'))
            source = 0 if broadcast_buffers else rank
            np.testing.assert_array_equal(previous.cpu().numpy(), pattern(source, max(0, index-1)))
            np.testing.assert_array_equal(previous_extra.cpu().numpy(),
                                          (np.arange(5)+source+max(0, index-1)) % 2 == 0)
            expected = index+1+rank if broadcast_buffers else (index+1)*(rank+1)
            assert offset.item() == expected
            np.testing.assert_array_equal(extra_offset.cpu().numpy(), np.full(3, 2*expected))
            assert np.isfinite(loss.item())
        assert runner.stats['captured'] == 1, runner.last_invalidated
        assert runner.stats['graph'] >= 5, runner._graph_refused
        assert offset_alias.item() == (12+rank if broadcast_buffers else 12*(rank+1))
        np.testing.assert_array_equal(mask_alias.cpu().numpy(), pattern(rank, 11))
        np.testing.assert_array_equal(extra_offset_alias.cpu().numpy(),
                                      np.full(3, 2*(12+rank if broadcast_buffers else 12*(rank+1))))
        np.testing.assert_array_equal(extra_mask_alias.cpu().numpy(), (np.arange(5)+rank+11) % 2 == 0)
        assert jt.core.backend_fallback_count() == before
    finally:
        runner.invalidate()
