"""Rank-local BatchNorm and DDP's in-place buffer broadcasts."""
import pytest
import numpy as np
import torch
import jittor as jt
from jittor.compat import collectives


class BufferedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((1,)))
        self.register_buffer('offset', torch.zeros((1,)))
        self.register_buffer('temporary', torch.zeros((1,)), persistent=False)

    def forward(self, value):
        return value * self.weight + self.offset


@pytest.mark.parametrize('broadcast_buffers', (True, False))
def test_ddp_broadcasts_buffers_before_forward_and_honors_the_option(monkeypatch, broadcast_buffers):
    module = BufferedModule()
    # Other suites deliberately reload the public modules. Patch the owners
    # used by this exported class, rather than a possibly replaced alias.
    namespace = torch.nn.parallel.DistributedDataParallel._jittor_ddp_broadcast_parameters.__globals__
    monkeypatch.setitem(namespace, '_ddp_world_size', lambda: 2)
    calls = []

    def broadcast(value):
        calls.append(value)
        with torch.no_grad():
            value.fill_(7.)
        return value

    monkeypatch.setattr(namespace['_collectives'], '_broadcast_from_rank0', broadcast)
    ddp = torch.nn.parallel.DistributedDataParallel(module, broadcast_buffers=broadcast_buffers)
    buffer_count = len(list(module.buffers()))
    assert len(calls) == len(list(module.parameters())) + (buffer_count if broadcast_buffers else 0)
    calls.clear()
    with torch.no_grad():
        module.offset.zero_()
        module.temporary.zero_()
    ddp(torch.ones((1,)))
    assert len(calls) == (buffer_count if broadcast_buffers else 0)
    np.testing.assert_array_equal(module.offset.numpy(), [7. if broadcast_buffers else 0.])


def test_rank0_broadcast_preserves_existing_storage_aliases(monkeypatch):
    value = torch.tensor([1., 2.])
    alias = value.detach()
    before = value.var_ptr
    monkeypatch.setattr(collectives, '_world_size', lambda: 2)
    monkeypatch.setattr(jt.Var, 'mpi_broadcast', lambda self: self + 3., raising=False)
    collectives._broadcast_from_rank0(value)
    assert value.var_ptr == before
    np.testing.assert_array_equal(alias.numpy(), [4., 5.])


def test_standard_batchnorm_is_local_and_explicit_sync_remains_available():
    for name in ('BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'):
        assert not getattr(torch.nn, name)(3).sync
    assert jt.nn.BatchNorm(3).sync
    assert torch.nn.SyncBatchNorm(3).sync
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
    converted = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    assert converted[0].sync
