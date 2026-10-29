"""Default Accelerate DDP hook import and explicit unsupported boundaries."""
import importlib

import pytest
import torch


def _accelerate_types():
    pytest.importorskip("accelerate")
    from accelerate.utils.dataclasses import (
        DDPCommunicationHookType, DistributedDataParallelKwargs,
    )
    return DDPCommunicationHookType, DistributedDataParallelKwargs


def test_comm_hook_modules_are_importable_and_declared_unimplemented():
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of
    prefix = "torch.distributed.algorithms.ddp_comm_hooks"
    package = importlib.import_module(prefix)
    defaults = importlib.import_module(prefix + ".default_hooks")
    power = importlib.import_module(prefix + ".powerSGD_hook")
    assert package.default_hooks is defaults
    assert package.powerSGD_hook is power
    for module, names in (
            (defaults, ("fp16_compress_hook", "bf16_compress_hook",
                        "fp16_compress_wrapper", "bf16_compress_wrapper")),
            (power, ("powerSGD_hook", "batched_powerSGD_hook", "PowerSGDState"))):
        for name in names:
            assert callable(getattr(module, name))
            assert fidelity_of(module.__name__ + "." + name).level == Fidelity.UNIMPLEMENTED
    assert fidelity_of(
        "torch.nn.parallel.DistributedDataParallel.register_comm_hook"
    ).level == Fidelity.UNIMPLEMENTED


def test_accelerate_default_comm_hook_does_not_register():
    _, kwargs_type = _accelerate_types()

    class Model:
        def register_comm_hook(self, *args, **kwargs):
            pytest.fail("default NO hook attempted registration")

    assert kwargs_type().register_comm_hook(Model()) is None


@pytest.mark.parametrize("hook_name", ["FP16", "BF16", "POWER_SGD", "BATCHED_POWER_SGD"])
def test_accelerate_selected_hook_rejected(hook_name):
    hook_type, kwargs_type = _accelerate_types()
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
    with pytest.raises(NotImplementedError, match="DDP.*(hook|state)"):
        kwargs_type(comm_hook=getattr(hook_type, hook_name)).register_comm_hook(model)


@pytest.mark.parametrize("wrapper_name", ["FP16", "BF16"])
def test_accelerate_selected_wrapper_rejected(wrapper_name):
    hook_type, kwargs_type = _accelerate_types()
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
    with pytest.raises(NotImplementedError, match="DDP.*wrapper"):
        kwargs_type(comm_hook=hook_type.FP16,
                    comm_wrapper=getattr(hook_type, wrapper_name)).register_comm_hook(model)


def test_ddp_rejects_custom_comm_hook_at_registration():
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 1))
    with pytest.raises(NotImplementedError, match="communication hooks"):
        model.register_comm_hook(None, lambda state, bucket: bucket)


@pytest.mark.parametrize("name", ["fp16_compress_hook", "bf16_compress_hook",
                                  "fp16_compress_wrapper", "bf16_compress_wrapper"])
def test_direct_default_hook_and_wrapper_calls_rejected(name):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    with pytest.raises(NotImplementedError, match="DDP"):
        getattr(default_hooks, name)(None)


@pytest.mark.parametrize("name", ["powerSGD_hook", "batched_powerSGD_hook", "PowerSGDState"])
def test_direct_power_sgd_calls_rejected(name):
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
    with pytest.raises(NotImplementedError, match="DDP"):
        getattr(powerSGD_hook, name)(None)
