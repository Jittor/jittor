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
