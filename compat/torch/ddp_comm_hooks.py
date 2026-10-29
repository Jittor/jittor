"""Import surface for DDP hooks whose execution is not yet supported."""
import types

from .fidelity import Fidelity, register_api_bindings
from ..transaction import active_transaction, set_attr


def fp16_compress_hook(*args, **kwargs):
    raise NotImplementedError("DDP FP16 communication hook is not supported")


def bf16_compress_hook(*args, **kwargs):
    raise NotImplementedError("DDP BF16 communication hook is not supported")


def fp16_compress_wrapper(*args, **kwargs):
    raise NotImplementedError("DDP FP16 communication wrapper is not supported")


def bf16_compress_wrapper(*args, **kwargs):
    raise NotImplementedError("DDP BF16 communication wrapper is not supported")


def powerSGD_hook(*args, **kwargs):
    raise NotImplementedError("DDP PowerSGD communication hook is not supported")


def batched_powerSGD_hook(*args, **kwargs):
    raise NotImplementedError("DDP batched PowerSGD communication hook is not supported")


class PowerSGDState:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DDP PowerSGD communication state is not supported")


def _publish_new_module(modules, name, module):
    """Record both the registry mapping and its publication bookkeeping."""
    transaction = active_transaction()
    if transaction is None:
        modules[name] = module
        return
    registry = getattr(modules, "registry", None)
    published = getattr(registry, "_published", None)
    before = dict(published) if published is not None else None
    try:
        transaction.publish_module(modules, name, module)
    finally:
        if before is not None:
            transaction.record_mapping_diffs(published, before)


def install_ddp_comm_hooks(modules, algorithms):
    """Publish names imported unconditionally by Accelerate's NO-hook path."""
    prefix = "torch.distributed.algorithms.ddp_comm_hooks"
    package = modules.get(prefix)
    new_package = package is None
    if new_package:
        package = types.ModuleType(prefix)
    if not hasattr(package, "__path__"):
        set_attr(package, "__path__", [])
    # The registry also binds parents when publishing: record the parent first
    # so that both a new publication and a reused module can be rolled back.
    set_attr(algorithms, "ddp_comm_hooks", package)
    if new_package:
        _publish_new_module(modules, prefix, package)
    for suffix, names in (
            ("default_hooks", ("fp16_compress_hook", "bf16_compress_hook",
                               "fp16_compress_wrapper", "bf16_compress_wrapper")),
            ("powerSGD_hook", ("powerSGD_hook", "batched_powerSGD_hook", "PowerSGDState"))):
        name = prefix + "." + suffix
        module = modules.get(name)
        new_module = module is None
        if new_module:
            module = types.ModuleType(name)
        set_attr(package, suffix, module)
        if new_module:
            _publish_new_module(modules, name, module)
        for symbol in names:
            set_attr(module, symbol, globals()[symbol])
        register_api_bindings(module, name, names, Fidelity.UNIMPLEMENTED,
                              "Import-compatible only; DDP communication hooks are unsupported")
