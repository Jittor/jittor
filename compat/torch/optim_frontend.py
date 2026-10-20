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


def _initialize_default(self, params, lr, args, kwargs, name):
    initializer = get_install_context(jt).state["optimizer_native_api"]["initializers"][name]
    return initializer(self, params, lr, *args, **kwargs)


def _validate_sgd_groups(self):
    defaults = {name: getattr(self, name) for name in (
        "lr", "momentum", "dampening", "weight_decay", "nesterov",
        "foreach", "fused", "maximize", "differentiable") if hasattr(self, name)}
    for group in self.param_groups:
        _validate_sgd_options(dict(defaults, **group))


def _validate_sgd_options(options):
    foreach, fused = options.get("foreach"), options.get("fused")
    for name, value in (("foreach", foreach), ("fused", fused)):
        if value is not None and not isinstance(value, bool):
            raise TypeError("%s must be bool or None" % name)
    for name in ("maximize", "differentiable"):
        value = options.get(name, False)
        if value:
            raise NotImplementedError("torch.optim.SGD(%s=True) is unsupported" % name)
    if foreach:
        raise NotImplementedError(
            "torch.optim.SGD(foreach=True) is unsupported; use foreach=False")
    lr, momentum = options.get("lr", .001), options.get("momentum", 0)
    dampening, weight_decay = options.get("dampening", 0), options.get("weight_decay", 0)
    if lr < 0 or momentum < 0 or weight_decay < 0:
        raise ValueError("SGD lr, momentum and weight_decay must be non-negative")
    if options.get("nesterov", False) and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    if dampening != 0:
        raise NotImplementedError(
            "torch.optim.SGD(dampening!=0) is unsupported; use dampening=0")


def initialize_sgd(self, params, lr=0.001, momentum=0, dampening=0,
                   weight_decay=0, nesterov=False, *, maximize=False,
                   foreach=None, differentiable=False, fused=None):
    _validate_sgd_options(dict(lr=lr, momentum=momentum, dampening=dampening,
                              weight_decay=weight_decay, nesterov=nesterov,
                              maximize=maximize, foreach=foreach,
                              differentiable=differentiable, fused=fused))
    result = _initialize_algorithm("SGD", self,
        (params, lr, momentum, weight_decay, dampening, nesterov), {"fused": fused})
    self.foreach, self.maximize, self.differentiable = foreach, maximize, differentiable
    for group in self.param_groups:
        for name in ("momentum", "dampening", "weight_decay", "nesterov",
                     "foreach", "fused", "maximize", "differentiable"):
            group.setdefault(name, getattr(self, name))
        _validate_sgd_options(group)
    return result


def initialize_adam(self, *args, **kwargs):
    return _initialize_algorithm("Adam", self, args, kwargs)


def initialize_adamw(self, *args, **kwargs):
    # The public Torch signature's fifth positional argument is weight_decay.
    foreach = kwargs.pop("foreach", None)
    if foreach is not None and not isinstance(foreach, bool):
        raise TypeError("foreach must be bool or None")
    if foreach is True:
        raise NotImplementedError(
            "torch.optim.AdamW(foreach=True) is not supported; use foreach=False"
        )
    if len(args) < 5 and "weight_decay" not in kwargs:
        kwargs["weight_decay"] = 0.01
    result = _initialize_algorithm("AdamW", self, args, kwargs)
    if foreach is not None:
        self.foreach = foreach
        for group in self.param_groups:
            group.setdefault("foreach", foreach)
    return result


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
