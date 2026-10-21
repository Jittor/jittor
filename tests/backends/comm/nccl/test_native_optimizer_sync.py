"""Actual MPI-free NCCL native optimizer synchronization, with and without DDP."""

import sys

import jittor as jt
from jittor import distributed as dist
from jittor.nn.parallel import DistributedDataParallel
from jittor._runtime.fallback import forbid_backend_fallbacks
import pytest

from _helpers import native_checkpoint as case


pytestmark = pytest.mark.skipif(
    not jt.compiler.has_cuda or not jt.compile_extern._jt_nccl_no_mpi,
    reason="requires an actual MPI-free NCCL launcher environment",
)


@pytest.mark.parametrize("ddp", [False, True])
def test_native_optimizer_nccl_mean_and_parameter_sync(monkeypatch, ddp):
    assert "torch" not in sys.modules and jt.flags.use_cuda
    world, rank = dist.get_world_size(), dist.get_rank()
    assert world in (1, 2)
    model, optimizer = case.model_optimizer()
    optimizer.param_sync_iter = 2
    if ddp:
        model = DistributedDataParallel(model)
    calls = {"gradients": 0, "broadcast": 0}
    original_reduce, original_broadcast = dist.all_reduce, dist.broadcast

    def reduce(tensor, op="mean", **kwargs):
        if tuple(tensor.shape) in ((2, 3), (2,)):
            assert op == "mean"
            calls["gradients"] += 1
        return original_reduce(tensor, op=op, **kwargs)

    def broadcast(tensor, src=0, **kwargs):
        calls["broadcast"] += 1
        return original_broadcast(tensor, src=src, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", reduce)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        reference = case.numpy_trajectory(5, world_size=world)
        for index in range(5):
            trace = case.training_step(model, optimizer, rank)
            case.assert_trace(trace, reference[index], rank=rank,
                              model_prefix="module." if ddp else "")
            for tensor in model.parameters():
                assert jt.core.dispatch_context([tensor]) == ("cuda", 0)
    # Each unmanaged gradient, or each multi-rank DDP-owned gradient, passes
    # the real owner exactly once. DDP never enters the legacy broadcast path.
    assert calls["gradients"] == (0 if ddp and world == 1 else 10)
    assert calls["broadcast"] == (0 if ddp else 4)
    assert jt.core.backend_fallback_count() == before == 0
    assert "torch" not in sys.modules
