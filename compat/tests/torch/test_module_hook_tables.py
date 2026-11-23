"""Private Torch hook tables reflect registered callbacks."""

import pytest
import torch


TABLES = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def test_fresh_modules_expose_separate_empty_hook_tables(device):
    a = torch.nn.Linear(2, 1, device=device)
    b = torch.nn.Linear(2, 1, device=device)
    for name in TABLES:
        table = getattr(a, name)
        assert not table
        assert table is getattr(a, name)
        assert table is not getattr(b, name)


def test_forward_tables_follow_hook_registration_and_removal(device):
    module = torch.nn.Linear(2, 1, bias=False, device=device)
    with torch.no_grad():
        module.weight.fill_(1.)
    before, after = module._forward_pre_hooks, module._forward_hooks
    pre = module.register_forward_pre_hook(lambda _, args: (args[0] + 1.,))
    post = module.register_forward_hook(lambda _, inputs, output: output * 2.)
    assert before is module._forward_pre_hooks and len(before) == 1
    assert after is module._forward_hooks and len(after) == 1
    value = torch.ones((1, 2), device=device)
    output = module(value)
    assert output.device.type == device and output.item() == 8.
    pre.remove()
    post.remove()
    assert not before and not after
    assert module(value).item() == 2.


def test_backward_table_follows_the_executed_callback(device):
    module = torch.nn.Linear(2, 1, bias=False, device=device)
    with torch.no_grad():
        module.weight.fill_(1.)
    calls = []
    table = module._backward_hooks

    def record(owner, inputs, outputs):
        calls.append((owner, outputs[0].device.type))

    handle = module.register_backward_hook(record)
    assert table is module._backward_hooks and len(table) == 1
    assert not module._backward_pre_hooks
    value = torch.ones((1, 2), device=device, requires_grad=True)
    module(value).sum().backward()
    assert calls == [(module, device)]
    torch.testing.assert_close(value.grad, torch.ones_like(value))
    handle.remove()
    assert not table
    module(torch.ones((1, 2), device=device, requires_grad=True)).sum().backward()
    assert calls == [(module, device)]
