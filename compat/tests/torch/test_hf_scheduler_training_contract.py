"""Transformers' linear warmup scheduler must advance on optimizer updates."""
import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup


def test_linear_warmup_schedule_advances_and_reaches_zero_after_updates():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1, num_training_steps=4)

    assert scheduler.last_epoch == 0
    rates_used = []
    with torch.no_grad():
        for _ in range(4):
            rates_used.append(float(scheduler.get_last_lr()[0]))
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()

    np.testing.assert_allclose(
        rates_used, [0.0, 3e-5, 2e-5, 1e-5], rtol=1e-12, atol=0.0)
    assert scheduler.last_epoch == 4
    assert float(scheduler.get_last_lr()[0]) == 0.0
