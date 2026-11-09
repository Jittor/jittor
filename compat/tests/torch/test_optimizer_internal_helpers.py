"""Eager optimizer internals consumed by third-party Adam implementations."""
import torch
from torch.optim.optimizer import (
    DeviceDict,
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_value,
    _stack_if_compiling,
    _view_as_real,
)


def test_optimizer_internal_helpers_select_supported_eager_path():
    parameter = torch.tensor([1.0], requires_grad=True)

    assert DeviceDict is dict
    assert _default_to_fused_or_foreach(
        [parameter], differentiable=False, use_fused=False
    ) == (False, False)
    assert "cuda" in _get_capturable_supported_devices()
    assert _get_value(torch.tensor(3.0)) == 3.0

    values = [1.0, 2.0]
    assert torch._utils.is_compiling() is False
    assert _stack_if_compiling(values) is values


def test_disable_dynamo_decorator_preserves_eager_callable():
    calls = []

    def single_tensor():
        return None

    @_disable_dynamo_if_unsupported(single_tensor_fn=single_tensor)
    def optimizer_step(value):
        calls.append(value)
        return value + 1

    assert optimizer_step(4) == 5
    assert calls == [4]


def test_view_as_real_leaves_real_optimizer_lists_unchanged():
    parameter = torch.tensor([1.0, 2.0])
    gradient = torch.tensor([3.0, 4.0])
    params = [parameter]
    grads = [gradient]

    _view_as_real(params, grads)

    assert params[0] is parameter
    assert grads[0] is gradient

def test_adam_parameter_groups_publish_torch_execution_defaults():
    parameter = torch.tensor([1.0], requires_grad=True)
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }

    adam = torch.optim.Adam([parameter])
    adamw = torch.optim.AdamW([parameter])

    for optimizer in (adam, adamw):
        group = optimizer.param_groups[0]
        for key, value in expected.items():
            assert group[key] == value
        assert group["lr"] == 1e-3
    assert adam.param_groups[0]["weight_decay"] == 0
    assert adam.param_groups[0]["decoupled_weight_decay"] is False
    assert adamw.param_groups[0]["weight_decay"] == 0.01
    assert adamw.param_groups[0]["decoupled_weight_decay"] is True
