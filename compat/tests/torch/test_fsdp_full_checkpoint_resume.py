"""Two-rank full model/optimizer checkpoint restores uneven FSDP shards."""
import copy
import gc
import os
from datetime import timedelta

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    os.environ.get('JITTOR_TORCH_FSDP2_DIM0_EXPERIMENT') != '1'
    or int(os.environ.get('WORLD_SIZE', '1')) != 2,
    reason='requires opt-in two-rank FSDP2',
)
def test_full_checkpoint_resume_with_frozen_snapshot():
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_state_dict, set_model_state_dict, set_optimizer_state_dict,
    )
    from torch.distributed.fsdp import fully_shard

    if os.environ.get('JITTOR_TORCH_SHIM') == '1':
        import jittor
        from pathlib import Path
        assert Path(jittor.__file__).resolve().is_relative_to(Path(os.environ['TRL_TEST_SOURCE']).resolve() / 'python/jittor')
    local_device = 0 if os.environ.get('JITTOR_TORCH_SHIM') == '1' else int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_device)
    dist = torch.distributed
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group('nccl', timeout=timedelta(seconds=45))
    try:
        rank = dist.get_rank()

        def make():
            torch.manual_seed(123)
            model = torch.nn.Linear(3, 5, bias=False).cuda()
            fully_shard(model)
            return model, torch.optim.AdamW(model.parameters(), lr=0.01)

        def step(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            loss = model(torch.ones((1, 3), device='cuda') * (rank + 1)).sum()
            torch.cuda.synchronize()
            dist.barrier()
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            torch.cuda.synchronize()

        model, optimizer = make()
        step(model, optimizer)
        snapshot = get_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        # On real Torch, optimizer-state tensors can alias the running optimizer.
        model_state, optimizer_state = copy.deepcopy(snapshot)
        saved_local = model.weight.to_local().detach().cpu().numpy().copy()
        dist.barrier()
        step(model, optimizer)
        expected = model.weight.to_local().detach().cpu().numpy().copy()
        dist.barrier()
        del model, optimizer
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()

        restored, restored_optimizer = make()
        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        # Torch 2.8's joint entry point skips the model on a rank with an empty
        # model dict. Separate public entry points let both ranks participate.
        set_optimizer_state_dict(restored, restored_optimizer, optimizer_state, options=options)
        set_model_state_dict(restored, model_state, options=options)
        torch.cuda.synchronize()
        np.testing.assert_array_equal(restored.weight.to_local().detach().cpu().numpy(), saved_local)
        dist.barrier()
        step(restored, restored_optimizer)
        np.testing.assert_array_equal(restored.weight.to_local().detach().cpu().numpy(), expected)
    finally:
        if initialized_here:
            dist.destroy_process_group()
