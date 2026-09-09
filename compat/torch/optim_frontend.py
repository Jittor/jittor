"""Independent optimizer types reusing native update implementations."""

from types import ModuleType, MappingProxyType
from collections.abc import Mapping
from .frontend import tensor_frontend
from .context import get_install_context
import jittor as jt


def initialize_base(self, params, lr, *args, **kwargs):
    state = get_install_context(jt).state["optimizer_frontend_native"]
    tensor_type = state["tensor_type"]
    if isinstance(params, tensor_type._frontend_backend.Var):
        raise TypeError("optimizer params must be an iterable of tensors or parameter groups")
    params = list(params)
    params = [dict(group, params=([group["params"]]
                                 if isinstance(group["params"], tensor_type._frontend_backend.Var)
                                 else list(group["params"])))
              if isinstance(group, Mapping) else group for group in params]
    with tensor_frontend(tensor_type):
        state["base_init"](self, params, lr, *args, **kwargs)


def _initialize_algorithm(name, instance, args, kwargs):
    state = get_install_context(jt).state["optimizer_frontend_native"]
    with tensor_frontend(state["tensor_type"]):
        state["algorithms"][name](instance, *args, **kwargs)


def initialize_sgd(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return _initialize_algorithm("SGD", self,
        (params, lr, momentum, weight_decay, dampening, nesterov), {})


def initialize_adam(self, *args, **kwargs):
    return _initialize_algorithm("Adam", self, args, kwargs)


def initialize_adamw(self, *args, **kwargs):
    return _initialize_algorithm("AdamW", self, args, kwargs)


def initialize_rmsprop(self, *args, **kwargs):
    return _initialize_algorithm("RMSprop", self, args, kwargs)


def initialize_adan(self, *args, **kwargs):
    return _initialize_algorithm("Adan", self, args, kwargs)


_ALGORITHM_INITIALIZERS = {
    "SGD": initialize_sgd, "Adam": initialize_adam, "AdamW": initialize_adamw,
    "RMSprop": initialize_rmsprop, "Adan": initialize_adan,
}


def make_optimizer_frontend(native, tensor_type):
    module = ModuleType("torch.optim")
    module.__package__ = "torch.optim"
    module.__path__ = []
    setattr(module, "_native_optimizer_module", native)
    get_install_context(tensor_type._frontend_backend).state["optimizer_frontend_native"] = MappingProxyType({
        "tensor_type": tensor_type,
        "base_init": native.Optimizer.__init__,
        "algorithms": MappingProxyType({name: getattr(native, name).__init__
                                       for name in _ALGORITHM_INITIALIZERS if hasattr(native, name)}),
    })

    base = type("Optimizer", (native.Optimizer,), {
        "__module__": "torch.optim",
        "__init__": initialize_base,
    })
    setattr(module, "Optimizer", base)
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan"):
        algorithm = getattr(native, name, None)
        if algorithm is None:
            continue
        # Native super() calls traverse this MRO through the frontend base,
        # whose state/closure adapters can be installed without changing the
        # original Optimizer or any native algorithm class dictionary.
        setattr(module, name, type(name, (algorithm, base), {
            "__module__": "torch.optim",
            "__init__": _ALGORITHM_INITIALIZERS[name],
        }))
    for name in ("opt_grad", "LRScheduler", "LambdaLR"):
        if hasattr(native, name):
            setattr(module, name, getattr(native, name))
    setattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    return module
