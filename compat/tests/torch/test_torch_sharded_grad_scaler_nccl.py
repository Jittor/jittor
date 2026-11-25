"""Real two-rank NCCL AMP overflow agreement.

Run via python/jittor/distributed/launch.py -n 2 --backend nccl --logdir ...
with JITTOR_TORCH_SHIM=1 and isolated rank caches.
"""
import os

import numpy as np
import pytest
import torch
import jittor as jt
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.skipif(os.environ.get("JT_NCCL_WORLD_SIZE") != "2",
                    reason="requires actual two-rank NCCL allocation")
def test_fp16_overflow_on_one_rank_skips_both_and_restores_scale():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    assert torch.distributed.get_world_size() == 2
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float16, device="cuda"))
        opt = torch.optim.SGD([p], lr=0.25)
        scaler = ShardedGradScaler("cuda", init_scale=8., growth_interval=1)
        for overflow, expected, scale in [(True, 1., 4.), (False, .75, 8.)]:
            opt.zero_grad()
            scaler.scale(p.float().sum()).backward()
            if overflow and rank == 1:
                p.grad.fill_(float("inf"))
            scaler.step(opt)
            scaler.update()
            p.sync()
            assert p.placement_backend == 1 and p.location() == "device"
            np.testing.assert_array_equal(p.numpy(), [expected])
            assert scaler.get_scale() == scale
        restored = ShardedGradScaler("cuda", init_scale=1.)
        restored.load_state_dict(scaler.state_dict())
        assert restored.state_dict() == scaler.state_dict()
