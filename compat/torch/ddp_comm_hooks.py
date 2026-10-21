"""Import graph for DDP communication hooks.

The default DDP path does not register a hook, but Accelerate eagerly imports
these names even with comm_hook=NO.  Publishing their names must not make
unsupported compression or PowerSGD silently act like an uncompressed all-reduce.
"""
from types import ModuleType


def _unsupported(*args, **kwargs):
    raise NotImplementedError(
        "Jittor torch DDP communication hooks are not implemented; use the default all-reduce"
    )


class PowerSGDState:
    def __init__(self, *args, **kwargs):
        _unsupported()


def install_ddp_comm_hooks(module_map, algorithms):
    root_name = "torch.distributed.algorithms.ddp_comm_hooks"
    root = module_map.setdefault(root_name, ModuleType(root_name))
    root.__path__ = []
    algorithms.ddp_comm_hooks = root
    for suffix, names in (
        ("default_hooks", (
            "fp16_compress_hook", "bf16_compress_hook",
            "fp16_compress_wrapper", "bf16_compress_wrapper",
        )),
        ("powerSGD_hook", ("powerSGD_hook", "batched_powerSGD_hook")),
    ):
        name = root_name + "." + suffix
        module = module_map.setdefault(name, ModuleType(name))
        for attr in names:
            setattr(module, attr, _unsupported)
        if suffix == "powerSGD_hook":
            module.PowerSGDState = PowerSGDState
        setattr(root, suffix, module)
