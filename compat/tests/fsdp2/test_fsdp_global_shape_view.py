"""Regression for the isolated Torch-FSDP global shape view experiment."""
from types import SimpleNamespace

import jittor as jt
import torch

from jittor.compat.fsdp2.shard import _install_global_shape_view


def test_public_shape_does_not_change_native_holder_or_parameter_identity():
    native = jt.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    identity = id(parameter)
    frontend = getattr(type(parameter), "_frontend_result_type", type(parameter))
    owner = torch.nn.Module()
    owner.register_parameter("weight", parameter)
    object.__setattr__(parameter, "_jittor_fsdp2_entry", SimpleNamespace(shape=(4, 3)))

    assert _install_global_shape_view(parameter) is parameter
    assert _install_global_shape_view(parameter) is parameter
    assert id(parameter) == identity and owner.weight is parameter
    assert getattr(type(parameter), "_frontend_result_type", type(parameter)) is frontend
    assert isinstance(parameter, torch.nn.Parameter)
    assert parameter.shape == torch.Size((4, 3))
    assert parameter.ndim == 2 and parameter.numel() == 12
    assert tuple(jt.Var.shape.__get__(parameter, type(parameter))) == (2, 3)
    assert tuple(native.shape) == (2, 3)
