"""Mutable-mapping behavior of the live optimizer state view."""
import torch


def test_optimizer_state_setdefault_returns_live_parameter_state():
    parameter = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    state = optimizer.state.setdefault(parameter, {})
    assert len(state) == 0

    step = torch.tensor(1.0)
    first = torch.tensor([3.0, 4.0])
    second = torch.tensor([5.0, 6.0])
    state["step"] = step
    state["exp_avg"] = first
    state["exp_avg_sq"] = second

    restored = optimizer.state.setdefault(parameter, {})
    assert restored["step"] is step
    assert restored["exp_avg"] is first
    assert restored["exp_avg_sq"] is second
