"""Optimizer updates preserve leaf identity, aliases and caller grad mode."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("name,options", [
    ("Adam", {}), ("AdamW", {}),
    ("AdamW", {"foreach": False, "fused": False}),
])
def test_adam_updates_preserve_parameter_leaf_and_live_alias(device, name, options):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    alias = parameter.detach()
    pointer = parameter.data_ptr()
    optimizer = getattr(torch.optim, name)([parameter], lr=.01, **options)
    closure_modes = []

    def closure():
        closure_modes.append(torch.is_grad_enabled())
        optimizer.zero_grad()
        loss = (parameter * torch.tensor([.25, -.5], device=device)).sum()
        loss.backward()
        return loss

    for _ in range(2):
        optimizer.step(closure)
        assert torch.is_grad_enabled() and parameter.is_leaf
        assert parameter.requires_grad and parameter.grad_fn is None
        assert parameter.data_ptr() == pointer
        np.testing.assert_array_equal(alias.cpu().numpy(), parameter.detach().cpu().numpy())
        assert not np.array_equal(parameter.detach().cpu().numpy(), [1., -2.])
        for key in ("exp_avg", "exp_avg_sq"):
            value = optimizer.state[parameter][key]
            assert not value.requires_grad and value.is_leaf and value.grad_fn is None
        parameter.requires_grad_(False)
        assert not parameter.requires_grad and parameter.is_leaf
        parameter.requires_grad_(True)
    assert closure_modes == [True, True]
