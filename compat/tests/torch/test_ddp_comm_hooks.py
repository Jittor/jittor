"""Default Accelerate DDP must not request unavailable communication hooks."""
from unittest import mock
import importlib
import sys
import types

import pytest
import torch


PREFIX = 'torch.distributed.algorithms.ddp_comm_hooks'


def test_accelerate_default_no_hook_registers_nothing():
    from accelerate.utils import DistributedDataParallelKwargs

    wrapper = mock.Mock()
    DistributedDataParallelKwargs().register_comm_hook(wrapper)
    wrapper.register_comm_hook.assert_not_called()


def test_hook_modules_have_canonical_import_identity_and_explicit_fidelity():
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of

    package = importlib.import_module(PREFIX)
    assert package is torch.distributed.algorithms.ddp_comm_hooks
    assert package.__spec__.submodule_search_locations is not None
    for suffix, names in (
        ('default_hooks', ('fp16_compress_hook', 'bf16_compress_hook',
                           'fp16_compress_wrapper', 'bf16_compress_wrapper')),
        ('powerSGD_hook', ('PowerSGDState', 'powerSGD_hook', 'batched_powerSGD_hook')),
    ):
        module = importlib.import_module(PREFIX + '.' + suffix)
        assert module is getattr(package, suffix) is sys.modules[module.__name__]
        for name in names:
            function = getattr(module, name)
            assert callable(function)
            record = fidelity_of(module.__name__ + '.' + name)
            assert record.level is Fidelity.UNIMPLEMENTED and record.implementation is function
            with pytest.raises(NotImplementedError, match='not implemented'):
                function(None, None)


@pytest.mark.parametrize('selection', ['FP16', 'BF16', 'POWER_SGD', 'BATCHED_POWER_SGD'])
def test_selected_accelerate_hooks_refuse_before_changing_reduction(selection):
    from accelerate.utils import DistributedDataParallelKwargs
    from accelerate.utils.dataclasses import DDPCommunicationHookType

    wrapper = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    state = wrapper._jittor_ddp_state
    kwargs = DistributedDataParallelKwargs(comm_hook=getattr(DDPCommunicationHookType, selection))
    with pytest.raises(NotImplementedError, match='not implemented'):
        kwargs.register_comm_hook(wrapper)
    assert wrapper._jittor_ddp_state is state
    assert wrapper.require_backward_grad_sync


@pytest.mark.parametrize('selection', ['FP16', 'BF16'])
def test_selected_compression_wrapper_refuses(selection):
    from accelerate.utils import DistributedDataParallelKwargs
    from accelerate.utils.dataclasses import DDPCommunicationHookType

    kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16,
        comm_wrapper=getattr(DDPCommunicationHookType, selection))
    wrapper = mock.Mock()
    with pytest.raises(NotImplementedError, match='not implemented'):
        kwargs.register_comm_hook(wrapper)
    wrapper.register_comm_hook.assert_not_called()


def test_arbitrary_custom_hook_registration_refuses_explicitly():
    from jittor.compat.torch.fidelity import Fidelity, fidelity_of

    wrapper = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    with pytest.raises(NotImplementedError, match='custom DDP communication hooks'):
        wrapper.register_comm_hook(None, lambda state, bucket: None)
    assert fidelity_of('torch.nn.parallel.DistributedDataParallel.register_comm_hook').level is Fidelity.UNIMPLEMENTED


def test_registry_namespace_publication_rolls_back_with_active_transaction():
    from jittor.compat.torch.context import ModuleRegistry
    from jittor.compat.torch.ddp_comm_hooks import install_ddp_comm_hooks
    from jittor.compat.transaction import ActivationTransaction

    owner = types.ModuleType('torch')
    distributed = types.ModuleType('torch.distributed')
    algorithms = types.ModuleType('torch.distributed.algorithms')
    owner.distributed = distributed
    distributed.algorithms = algorithms
    modules = {'torch': owner, 'torch.distributed': distributed,
               'torch.distributed.algorithms': algorithms}
    original = dict(modules)
    registry = ModuleRegistry(owner, modules)
    transaction = ActivationTransaction(owner)
    transaction.acquire()
    try:
        install_ddp_comm_hooks(registry)
        package = modules[PREFIX]
        assert algorithms.ddp_comm_hooks is package
        assert modules[PREFIX + '.default_hooks'] is package.default_hooks
        # The same installation transaction also handles a later failed step.
        transaction.rollback()
    finally:
        if transaction.state == 'open':
            transaction.rollback()
        transaction.release()
    assert modules == original
    assert not hasattr(algorithms, 'ddp_comm_hooks')
    assert registry._published == {}
