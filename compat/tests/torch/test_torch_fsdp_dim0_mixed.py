"""Two-rank regression for FSDP and ordinary parameters in one optimizer.

Run with the Torch shim (or genuine Torch oracle), NCCL, two allocated CUDA
devices and JITTOR_TORCH_FSDP2_DIM0_EXPERIMENT=1. The test is intentionally
skipped outside this explicitly selected distributed configuration.
"""

import os

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("JITTOR_TORCH_FSDP2_DIM0_EXPERIMENT") != "1"
    or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires opt-in two-rank FSDP test",
)
def test_mixed_sharded_and_plain_gradients_accumulate_and_clip():
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist = torch.distributed
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        model = torch.nn.Linear(3, 4, bias=False).cuda()
        fully_shard(model)
        parameter = model.weight
        initial = parameter.full_tensor().detach().cpu().numpy().copy()
        plain = torch.nn.Parameter(torch.ones((2,), device="cuda"))
        params = list(model.parameters()) + [plain]
        # Torch CUDA foreach cannot update plain Tensor and DTensor together
        # within one group; separate groups remain one mixed optimizer.
        optimizer = torch.optim.SGD([{"params": [parameter]}, {"params": [plain]}], lr=0.01)
        x = torch.ones((1, 3), device="cuda") * (rank + 1)

        def backward():
            (model(x).sum() + plain.sum() * 2).backward()

        def check(expected_shard, expected_plain):
            assert isinstance(parameter.grad, DTensor)
            assert not isinstance(plain.grad, DTensor)
            np.testing.assert_allclose(
                parameter.grad.full_tensor().detach().cpu().numpy(),
                expected_shard, rtol=1e-6, atol=1e-6,
            )
            np.testing.assert_allclose(
                plain.grad.detach().cpu().numpy(),
                expected_plain, rtol=1e-6, atol=1e-6,
            )

        optimizer.zero_grad(set_to_none=True)
        backward()
        check(1.5, 2.0)
        backward()
        check(3.0, 4.0)
        optimizer.zero_grad(set_to_none=True)
        assert parameter.grad is None and plain.grad is None
        backward()
        check(1.5, 2.0)

        optimizer.zero_grad(set_to_none=False)
        check(0.0, 0.0)
        backward()
        check(1.5, 2.0)

        sharded_norm = torch.nn.utils.get_total_norm([parameter.grad])
        assert isinstance(sharded_norm, DTensor)
        sharded_norm = sharded_norm.full_tensor()
        for order, expected in ((1, 18.0), (float("inf"), 1.5)):
            value = torch.nn.utils.get_total_norm([parameter.grad], norm_type=order)
            assert isinstance(value, DTensor)
            np.testing.assert_allclose(float(value.full_tensor().item()), expected, rtol=1e-6)
        plain_norm = torch.nn.utils.get_total_norm([plain.grad])
        total = torch.linalg.vector_norm(torch.stack([sharded_norm, plain_norm]), 2)
        np.testing.assert_allclose(float(total.item()), np.sqrt(35), rtol=1e-6)
        norm_on_mesh = DTensor.from_local(total, parameter.grad.device_mesh)
        torch.nn.utils.clip_grads_with_norm_([parameter], 1.0, norm_on_mesh)
        torch.nn.utils.clip_grads_with_norm_([plain], 1.0, total)
        check(1.5 / (np.sqrt(35) + 1e-6), 2.0 / (np.sqrt(35) + 1e-6))
        optimizer.step()
        np.testing.assert_allclose(
            parameter.full_tensor().detach().cpu().numpy(),
            initial - 0.01 * 1.5 / (np.sqrt(35) + 1e-6), rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            plain.detach().cpu().numpy(), 1.0 - 0.01 * 2.0 / (np.sqrt(35) + 1e-6),
            rtol=1e-6, atol=1e-6,
        )
    finally:
        if initialized_here:
            dist.destroy_process_group()
