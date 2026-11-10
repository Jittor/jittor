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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("failure", ["pre", "forward", "post"])
def test_nested_hook_failure_restores_outer_call_depth(device, failure):
    from jittor._core.module import Module

    require_rng_device(device)
    seen = []

    class Inner(torch.nn.Module):
        def forward(self, x):
            seen.append(("forward", Module._call_depth))
            if failure == "forward":
                raise ValueError("nested failure")
            return x * 2

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, x):
            assert Module._call_depth == 1
            try:
                self.inner(x)
            except ValueError as error:
                assert str(error) == "nested failure"
                # The failed child must restore the enclosing call, not zero.
                assert Module._call_depth == 1
            else:
                raise AssertionError("missing child error")
            return x + 3

    def pre(module, args):
        seen.append(("pre", Module._call_depth))
        if failure == "pre":
            raise ValueError("nested failure")

    def post(module, args, output):
        seen.append(("post", Module._call_depth))
        if failure == "post":
            raise ValueError("nested failure")

    model = Outer()
    model.inner.register_forward_pre_hook(pre)
    model.inner.register_forward_hook(post)
    with jt.flag_scope(auto_graph_replay=0), torch.no_grad():
        for value in (1., 2.):
            result = model(torch.tensor([value], device=device))
            assert result.device.type == device and result.item() == value + 3
            assert Module._call_depth == 0
    assert seen and all(depth == 2 for _, depth in seen)
