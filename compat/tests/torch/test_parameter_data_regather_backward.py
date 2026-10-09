"""A parameter retains its gradient after ZeRO-style data partition and regather."""

import torch


def test_parameter_data_regather_keeps_original_forward_gradient():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))
    inputs = torch.tensor([4.0, 5.0], dtype=torch.float32, requires_grad=True)
    events = []
    parameter.register_post_accumulate_grad_hook(
        lambda tensor: events.append(None if tensor.grad is None else tensor.grad.tolist())
    )

    loss = (parameter * inputs).sum()
    parameter.data = torch.empty(0, dtype=torch.float32)
    parameter.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
    loss.backward()

    assert inputs.grad.tolist() == [2.0, 3.0]
    assert parameter.grad is not None
    assert parameter.grad.tolist() == [4.0, 5.0]
    assert events == [[4.0, 5.0]]

    parameter.grad = None
    inputs.grad = None
    (parameter * inputs).sum().backward()
    assert inputs.grad.tolist() == [2.0, 3.0]
    assert parameter.grad.tolist() == [4.0, 5.0]
    assert events == [[4.0, 5.0], [4.0, 5.0]]


def test_regathered_parameter_can_use_current_full_node():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))
    inputs = torch.tensor([4.0, 5.0], dtype=torch.float32, requires_grad=True)
    events = []
    parameter.register_post_accumulate_grad_hook(
        lambda tensor: events.append(None if tensor.grad is None else tensor.grad.tolist())
    )

    parameter.data = torch.empty(0, dtype=torch.float32)
    parameter.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
    (parameter * inputs).sum().backward()

    assert inputs.grad.tolist() == [2.0, 3.0]
    assert parameter.grad is not None
    assert parameter.grad.tolist() == [4.0, 5.0]
    assert events == [[4.0, 5.0]]


def test_regathered_parameter_accumulates_multiple_forward_nodes():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))
    first = torch.tensor([4.0, 5.0], dtype=torch.float32, requires_grad=True)
    second = torch.tensor([6.0, 7.0], dtype=torch.float32, requires_grad=True)
    events = []
    parameter.register_post_accumulate_grad_hook(
        lambda tensor: events.append(None if tensor.grad is None else tensor.grad.tolist())
    )

    first_loss = (parameter * first).sum()
    parameter.data = torch.empty(0, dtype=torch.float32)
    parameter.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
    second_loss = (parameter * second).sum()
    parameter.data = torch.empty(0, dtype=torch.float32)
    parameter.data = torch.tensor([2.0, 3.0], dtype=torch.float32)
    (first_loss + second_loss).backward()

    assert first.grad.tolist() == [2.0, 3.0]
    assert second.grad.tolist() == [2.0, 3.0]
    assert parameter.grad is not None
    assert parameter.grad.tolist() == [10.0, 12.0]
    assert events == [[10.0, 12.0]]
