"""Module calls preserve argument identity, Python errors and context scope."""
import jittor as jt
import pytest
import torch

from _rng_init_support import preserve_random_streams, require_rng_device


@pytest.mark.parametrize("count", [0, 1, 7, 8, 32])
def test_forward_argument_identity_and_keyword_order(count):
    class Echo(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return args, kwargs

    values = tuple(object() for _ in range(count))
    keywords = {"tensor": torch.ones(1), "marker": object()}
    actual_args, actual_kwargs = Echo()(*values, **keywords)
    assert len(actual_args) == count
    assert all(left is right for left, right in zip(actual_args, values))
    assert list(actual_kwargs) == list(keywords)
    assert all(actual_kwargs[name] is value for name, value in keywords.items())


def test_nested_exception_restores_native_construction_type():
    before = type(jt.zeros(1))

    class Inner(torch.nn.Module):
        def forward(self, value, *, fail=False):
            if fail:
                raise ValueError("forward failure")
            return value + 1

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, value, *, fail=False):
            return self.inner(value, fail=fail)

    model = Outer()
    value = torch.ones(1)
    with pytest.raises(ValueError, match="forward failure"):
        model(value, fail=True)
    assert type(jt.zeros(1)) is before
    assert model(value).item() == 2
    with pytest.raises(TypeError):
        model(value, value=value)
    assert type(jt.zeros(1)) is before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dispatch_reads_live_weights_hooks_and_forward(device):
    require_rng_device(device)
    with preserve_random_streams(), torch.no_grad(), jt.flag_scope(auto_graph_replay=0):
        layer = torch.nn.Linear(2, 1, bias=False).to(device)
        model = torch.nn.Sequential(layer).eval()
        value = torch.tensor([[2., 3.]], device=device)

        def check(expected):
            result = model(value)
            assert result.device.type == device
            assert result.item() == expected

        layer.weight.fill_(1)
        check(5)
        handle = layer.register_forward_hook(lambda module, args, result: result + 7)
        check(12)
        handle.remove()
        layer.weight.fill_(2)
        check(10)
        layer.forward = lambda x: x.sum(dim=-1, keepdim=True) * 3
        check(15)
