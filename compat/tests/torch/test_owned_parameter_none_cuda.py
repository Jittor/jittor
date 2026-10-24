"""Owned Torch parameter declaration metadata, exercised by actual CUDA math."""
from _helpers import capability as _test_capability
import pytest


def _cuda_runtime():
    import jittor as jt
    if not _test_capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("accelerator prerequisite: CUDA runtime unavailable")
    return jt.flag_scope(use_cuda=1, auto_flush_ops=0, backend_fallback="error")


def _cuda_backward(module, shape):
    import jittor as jt
    import torch
    x = torch.ones(shape, device="cuda:0", requires_grad=True)
    y = module(x)
    y.square().sum().backward()
    torch.cuda.synchronize()
    parameters = list(module.parameters())
    assert parameters
    for value in [x, y, x.grad] + parameters + [p.grad for p in parameters]:
        assert value is not None
        assert value.device.type == "cuda" and value.device.index == 0
        assert torch.isfinite(value).all().item()
        assert value.location() == "device"
    assert jt.core.backend_fallback_count() == 0


@pytest.mark.parametrize("kind", ["Linear", "Conv1d"])
@pytest.mark.parametrize("bias", [False, True])
def test_owned_disabled_bias_declaration_and_cuda_backward(kind, bias):
    import torch
    with _cuda_runtime(), torch.device("cuda:0"):
        if kind == "Linear":
            module = torch.nn.Linear(3, 4, bias=bias)
            shape = (2, 3)
        else:
            module = torch.nn.Conv1d(3, 4, 3, bias=bias)
            shape = (2, 3, 5)
        parameters = module._parameters
        assert list(parameters) == ["weight", "bias"]
        assert parameters["weight"] is module.weight
        assert parameters["bias"] is module.bias
        assert (parameters["bias"] is None) is (not bias)
        assert list(dict(module.named_parameters(remove_duplicate=False))) == (
            ["weight", "bias"] if bias else ["weight"])
        _cuda_backward(module, shape)


def test_owned_explicit_none_order_alias_refresh_and_cuda_backward():
    import torch

    class Explicit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("none_first", None)
            self.register_parameter("none_second", None)
            self.ordinary_none = None
            self.weight = torch.nn.Parameter(torch.ones((3, 3)))
            self.weight_alias = self.weight

        def forward(self, x):
            return x @ self.weight + x @ self.weight_alias

    with _cuda_runtime(), torch.device("cuda:0"):
        module = Explicit()
        names = ["none_first", "none_second", "weight", "weight_alias"]
        assert list(module._parameters) == names
        assert module._parameters["none_first"] is None
        assert module._parameters["none_second"] is None
        assert "ordinary_none" not in module._parameters
        assert module._parameters["weight"] is module._parameters["weight_alias"]
        assert list(dict(module.named_parameters(remove_duplicate=False))) == names[2:]
        _cuda_backward(module, (2, 3))
        # Each new get must reflect actual current objects, not a cached map.
        replacement = torch.nn.Parameter(torch.ones((3, 3)))
        module.none_first = replacement
        assert list(module._parameters) == names
        assert module._parameters["none_first"] is replacement
        module.none_first = None
        assert list(module._parameters) == names
        assert module._parameters["none_first"] is None
        # The native write-through mapping implementation remains in use.
        view = module._parameters
        view["none_second"] = replacement
        assert module.none_second is replacement
        assert module._parameters["none_second"] is replacement
        view["none_second"] = None
        assert module.none_second is None
        assert module._parameters["none_second"] is None
        del view["none_second"]
        assert not hasattr(module, "none_second")
        assert list(module._parameters) == ["none_first", "weight", "weight_alias"]
        # Existing container/FSDP overrides remain the exact owner mapping.
        override = {"external": replacement}
        module._parameters = override
        assert module._parameters is override


def test_owned_parameter_container_preserves_native_keys():
    import jittor as jt
    import torch
    with _cuda_runtime(), torch.device("cuda:0"):
        p = torch.nn.Parameter(torch.ones((2, 2)))
        for container, names in ((torch.nn.ParameterList([p, p]), ["0", "1"]),
                                 (torch.nn.ParameterDict({"left": p, "right": p}),
                                  ["left", "right"])):
            assert list(container._parameters) == names
            assert all(value is p for value in container._parameters.values())
        (p.square().sum()).backward()
        torch.cuda.synchronize()
        assert p.grad.device.type == "cuda"
        assert torch.isfinite(p.grad).all().item()
        assert jt.core.backend_fallback_count() == 0
