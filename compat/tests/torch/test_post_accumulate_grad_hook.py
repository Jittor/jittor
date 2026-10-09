"""Post-accumulate tensor hooks needed by ZeRO gradient partitioning."""

import pytest
import torch


def test_hook_sees_accumulated_parameter_gradient_and_can_be_removed():
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    seen = []
    handle = parameter.register_post_accumulate_grad_hook(
        lambda tensor: seen.append((tensor is parameter, float(tensor.grad.item())))
    )

    (parameter * parameter).sum().backward()
    (parameter * parameter).sum().backward()
    assert seen == [(True, pytest.approx(4.0)), (True, pytest.approx(8.0))]
    assert float(parameter.grad.item()) == pytest.approx(8.0)

    handle.remove()
    handle.remove()
    optimizer.zero_grad()
    (parameter * parameter).sum().backward()
    assert len(seen) == 2


def test_hook_checks_leaf_grad_and_return_value():
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(TypeError, match="callable"):
        parameter.register_post_accumulate_grad_hook(3)
    with pytest.raises(RuntimeError, match="non-leaf"):
        (parameter * 2).register_post_accumulate_grad_hook(lambda _: None)
    frozen = torch.tensor([1.0], requires_grad=False)
    with pytest.raises(RuntimeError, match="doesn't require gradient"):
        frozen.register_post_accumulate_grad_hook(lambda _: None)

    handle = parameter.register_post_accumulate_grad_hook(lambda _: 1)
    with pytest.raises(RuntimeError, match="return None"):
        (parameter * parameter).sum().backward()
    handle.remove()