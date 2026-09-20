"""CUDA FSDP2 placement regression with the real two-rank NCCL launcher.

Run with the Jittor environment and NCCL configured:
    python python/jittor/distributed/launch.py -n 2 --backend nccl \
        --logdir <isolated-logdir> -- python -m pytest -q \
        compat/tests/torch/test_torch_fsdp2_cuda_placement.py
"""

import os

import pytest
import jittor as jt
import torch
from torch.distributed.fsdp import fully_shard


@pytest.mark.skipif(
    os.environ.get("JT_NCCL_WORLD_SIZE") != "2",
    reason="requires actual two-rank NCCL; single-process CPU cannot test placement",
)
def test_fully_shard_places_only_newly_managed_state_on_cuda():
    assert jt.world_size == 2 and torch.cuda.is_available()

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(2, 2, bias=False).cpu()
            self.trainable = torch.nn.Parameter(torch.ones(2, device="cpu"))
            self.frozen = torch.nn.Parameter(
                torch.ones(2, device="cpu"), requires_grad=False)
            self.ignored = torch.nn.Parameter(
                torch.ones(2, device="cpu"), requires_grad=False)
            self.register_buffer("scale", torch.ones(2, device="cpu"))

        def forward(self, x):
            return self.inner(x) * self.scale + self.trainable + self.frozen

    with jt.flag_scope(use_cuda=1):
        model = Nested()
        ignored, scale = model.ignored, model.scale
        assert all(p.device.type == "cpu" for p in model.parameters())
        fallback_before = int(jt.core.backend_fallback_count())
        fully_shard(model.inner)
        child_shard = model.inner.weight
        fully_shard(model, ignored_params=(ignored,))

        assert model.inner.weight is child_shard
        assert model.ignored is ignored and ignored.device.type == "cpu"
        assert model.scale is scale and scale.device.type == "cuda"
        assert model.trainable.requires_grad and not model.frozen.requires_grad
        assert model.trainable.device.type == "cuda"
        assert model.frozen.device.type == "cuda"
        assert child_shard.device.type == "cuda"

        optimizer = torch.optim.SGD([model.trainable, child_shard], lr=0.05)
        before = model.trainable.detach().clone()
        optimizer.zero_grad()
        loss = model(torch.ones((2, 2), device="cuda")).sum()
        loss.backward()
        assert model.trainable.grad is not None
        assert child_shard.grad is not None
        assert model.frozen.grad is None
        optimizer.step()
        jt.sync_all(True)
        local_delta = (float(torch.abs(model.trainable - before).max().item())
                       if model.trainable.numel() else 0.0)
        global_delta = jt.array([local_delta]).cuda().mpi_all_reduce("sum")
        assert float(global_delta.item()) > 0
        assert int(jt.core.backend_fallback_count()) == fallback_before
