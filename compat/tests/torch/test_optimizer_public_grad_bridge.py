"""AdamW must consume the public gradient after ZeRO replaces a parameter group."""
import pytest
import torch


def test_adamw_uses_public_gradient_when_internal_group_cache_is_absent():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
    parameter.grad = torch.tensor([1.0, 1.0])
    # DeepSpeed ZeRO owns and rebuilds optimizer groups. The public .grad is
    # authoritative even if a framework-specific per-group cache is absent.
    optimizer.param_groups[0].pop("grads", None)
    optimizer.step()
    assert abs(float(parameter[0].item()) - 0.9) < 1e-5
    assert abs(float(parameter[1].item()) - 1.9) < 1e-5


def test_adamw_replaces_stale_group_gradient_after_zero_rebinds_group():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    first_value = float(parameter.item())

    # ZeRO removes the flat parameter between steps. Its next public grad is
    # assigned while the base optimizer group is empty, leaving the old
    # group-local gradient at the same shape.
    group = optimizer.param_groups[0]
    group["params"] = []
    parameter.grad = torch.tensor([-10.0])
    group["params"] = [parameter]
    optimizer.step()

    assert first_value == pytest.approx(0.9)
    assert float(parameter.item()) == pytest.approx(0.9673807621, abs=1e-5)
