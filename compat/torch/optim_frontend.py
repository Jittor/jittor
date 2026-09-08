"""Independent optimizer types reusing native update implementations."""

from types import ModuleType
from functools import wraps
from collections.abc import Mapping
from .frontend import tensor_frontend


def make_optimizer_frontend(native, tensor_type):
    module = ModuleType("torch.optim")
    module.__package__ = "torch.optim"
    module.__path__ = []
    module._native_optimizer_module = native
    def frontend_init(implementation):
        @wraps(implementation)
        def initialize(self, *args, **kwargs):
            with tensor_frontend(tensor_type):
                implementation(self, *args, **kwargs)
        return initialize

    def initialize_base(self, params, lr, *args, **kwargs):
        if isinstance(params, tensor_type._frontend_backend.Var):
            raise TypeError("optimizer params must be an iterable of tensors or parameter groups")
        params = list(params)
        params = [dict(group, params=([group["params"]]
                                     if isinstance(group["params"], tensor_type._frontend_backend.Var)
                                     else list(group["params"])))
                  if isinstance(group, Mapping) else group for group in params]
        with tensor_frontend(tensor_type):
            native.Optimizer.__init__(self, params, lr, *args, **kwargs)

    base = type("Optimizer", (native.Optimizer,), {
        "__module__": "torch.optim",
        "__init__": initialize_base,
    })
    module.Optimizer = base
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan"):
        algorithm = getattr(native, name, None)
        if algorithm is None:
            continue
        # Native super() calls traverse this MRO through the frontend base,
        # whose state/closure adapters can be installed without changing the
        # original Optimizer or any native algorithm class dictionary.
        setattr(module, name, type(name, (algorithm, base), {
            "__module__": "torch.optim",
            "__init__": frontend_init(algorithm.__init__),
        }))
    for name in ("opt_grad", "LRScheduler", "LambdaLR"):
        if hasattr(native, name):
            setattr(module, name, getattr(native, name))
    module.__all__ = [name for name in vars(module) if not name.startswith("_")]
    return module
