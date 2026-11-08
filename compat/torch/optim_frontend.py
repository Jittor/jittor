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
    if foreach and fused:
        raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
    lr, momentum = options.get("lr", .001), options.get("momentum", 0)
    dampening, weight_decay = options.get("dampening", 0), options.get("weight_decay", 0)
    if lr < 0 or momentum < 0 or weight_decay < 0:
        raise ValueError("SGD lr, momentum and weight_decay must be non-negative")
    if options.get("nesterov", False) and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")


def initialize_sgd(self, params, lr=0.001, momentum=0, dampening=0,
                   weight_decay=0, nesterov=False, *, maximize=False,
                   foreach=None, differentiable=False, fused=None):
    _validate_sgd_options(dict(lr=lr, momentum=momentum, dampening=dampening,
                              weight_decay=weight_decay, nesterov=nesterov,
                              maximize=maximize, foreach=foreach,
                              differentiable=differentiable, fused=fused))
    result = _initialize_algorithm("SGD", self,
        (params, lr, momentum, weight_decay, dampening, nesterov), {"torch_math": True, "fused": fused})
    self.foreach, self.maximize, self.differentiable = foreach, maximize, differentiable
    for group in self.param_groups:
        for name in ("momentum", "dampening", "weight_decay", "nesterov",
                     "foreach", "fused", "maximize", "differentiable"):
            group.setdefault(name, getattr(self, name))
        _validate_sgd_options(group)
    return result


def _validate_adam_options(options):
    for name in ("foreach", "fused"):
        value = options.get(name)
        if value is not None and not isinstance(value, bool):
            raise TypeError("Adam %s must be bool or None" % name)
        if value is True:
            raise NotImplementedError(
                "torch.optim.Adam(%s=True) is unsupported; use False or None" % name)


def _validate_adam_groups(self):
    defaults = {name: getattr(self, name, None) for name in ("foreach", "fused")}
    for group in self.param_groups:
        _validate_adam_options(dict(defaults, **group))


def _restore_torch_moment_dtypes(groups):
    """Torch moments follow parameter dtype; native adaptive state stays FP32."""
    from jittor._core.dtypes import dtype_name
    with jt.flag_scope(amp_reg=0):
        for group in groups:
            for key in ("m", "values"):
                for index, parameter in enumerate(group["params"]):
                    buffer = group[key][index]
                    if dtype_name(buffer.dtype) != dtype_name(parameter.dtype):
                        group[key][index] = buffer.cast(parameter.dtype).stop_grad()


def add_adam_param_group(self, group):
    if not isinstance(group, Mapping):
        raise TypeError("param_group must be a mapping")
    group = dict(group)
    for name in ("foreach", "fused"):
        group.setdefault(name, getattr(self, name, None))
    _validate_adam_options(group)
    state = get_install_context(jt).state["optimizer_frontend_native"]
    var_type = state["tensor_type"]._frontend_backend.Var
    group["params"] = ([group["params"]] if isinstance(group["params"], var_type)
                       else list(group["params"]))
    with tensor_frontend(state["tensor_type"]):
        result = state["adam_add_param_group"](self, group)
        _restore_torch_moment_dtypes(self.param_groups[-1:])
        return result


def initialize_adam(self, *args, **kwargs):
    options = {name: kwargs.pop(name, None) for name in ("foreach", "fused")}
    _validate_adam_options(options)
    result = _initialize_algorithm("Adam", self, args, kwargs)
    _restore_torch_moment_dtypes(self.param_groups)
    self.foreach, self.fused = options["foreach"], options["fused"]
    for group in self.param_groups:
        for name, value in options.items():
            group.setdefault(name, value)
    _validate_adam_groups(self)
    return result


def _validate_adamw_capturable(value):
    if not isinstance(value, bool):
        raise TypeError("capturable must be bool")
    if value:
        raise NotImplementedError(
            "torch.optim.AdamW(capturable=True) is unsupported; use capturable=False")


def _validate_adamw_options(options):
    _validate_adamw_capturable(options.get("capturable", False))
    for name in ("foreach", "fused"):
        value = options.get(name)
        if value is not None and not isinstance(value, bool):
            raise TypeError("%s must be bool or None" % name)
    for name in ("amsgrad", "maximize", "differentiable"):
        value = options.get(name, False)
        if not isinstance(value, bool):
            raise TypeError("%s must be bool" % name)
        if value:
            raise NotImplementedError(
                "torch.optim.AdamW(%s=True) is unsupported" % name)
    if options.get("foreach") and options.get("fused"):
        raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
    if options.get("decoupled_weight_decay", True) is not True:
        raise NotImplementedError(
            "torch.optim.AdamW(decoupled_weight_decay=False) is unsupported")


def _validate_adamw_capturable_groups(self):
    # Keep the step adapter's single preflight before any state or parameter
    # update, including after callers mutate public options or restore groups.
    _validate_adamw_capturable(getattr(self, "capturable", False))
    for group in self.param_groups:
        _validate_adamw_options(group)


def _adamw_defaults(self):
    return self._torch_adamw_defaults


def _adamw_complete_group(group, defaults):
    group = dict(group)
    for name, value in defaults.items():
        group.setdefault(name, value)
    _validate_adamw_options(group)
    return group


def add_adamw_param_group(self, group):
    if not isinstance(group, Mapping):
        raise TypeError("param_group must be a mapping")
    group = _adamw_complete_group(group, self.defaults)
    state = get_install_context(jt).state["optimizer_frontend_native"]
    var_type = state["tensor_type"]._frontend_backend.Var
    group["params"] = ([group["params"]] if isinstance(group["params"], var_type)
                       else list(group["params"]))
    with tensor_frontend(state["tensor_type"]):
        result = state["adamw_add_param_group"](self, group)
        _restore_torch_moment_dtypes(self.param_groups[-1:])
        return result


def initialize_adamw(self, params, lr=1e-3, betas=(.9, .999), eps=1e-8,
                     weight_decay=.01, amsgrad=False, *, maximize=False,
                     foreach=None, capturable=False, differentiable=False,
                     fused=None):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                    amsgrad=amsgrad, maximize=maximize, foreach=foreach,
                    capturable=capturable, differentiable=differentiable,
                    fused=fused, decoupled_weight_decay=True)
    _validate_adamw_options(defaults)
    state = get_install_context(jt).state["optimizer_frontend_native"]
    if isinstance(params, state["tensor_type"]._frontend_backend.Var):
        raise TypeError("optimizer params must be an iterable of tensors or parameter groups")
    params = [_adamw_complete_group(group, defaults)
              if isinstance(group, Mapping) else group for group in params]
    self._torch_adamw_defaults = defaults
    # Native AdamW orders eps before betas; use names at this boundary.
    result = _initialize_algorithm("AdamW", self, (params, lr),
                                  dict(eps=eps, betas=betas,
                                       weight_decay=weight_decay, fused=fused))
    _restore_torch_moment_dtypes(self.param_groups)
    self.capturable, self.foreach = capturable, foreach
    for group in self.param_groups:
        for name, value in defaults.items():
            group.setdefault(name, value)
    return result


def initialize_rmsprop(self, *args, **kwargs):
    return _initialize_algorithm("RMSprop", self, args, kwargs)


def initialize_adan(self, *args, **kwargs):
    return _initialize_algorithm("Adan", self, args, kwargs)


def adagrad_defaults(self):
    """Torch exposes only the public hyperparameter defaults."""
    return self._adagrad_defaults


def initialize_adagrad(self, params, lr=1e-2, lr_decay=0, weight_decay=0,
                       initial_accumulator_value=0, eps=1e-10, foreach=None, *,
                       maximize=False, differentiable=False, fused=None):
    return _initialize_algorithm("Adagrad", self,
        (params, lr, lr_decay, weight_decay, initial_accumulator_value, eps, foreach),
        dict(maximize=maximize, differentiable=differentiable, fused=fused))


_ALGORITHM_INITIALIZERS = {
    "SGD": initialize_sgd, "Adam": initialize_adam, "AdamW": initialize_adamw,
    "RMSprop": initialize_rmsprop, "Adan": initialize_adan,
    "Adagrad": initialize_adagrad,
}


def make_optimizer_frontend(native, tensor_type):
    module = ModuleType("torch.optim")
    module.__package__ = "torch.optim"
    module.__path__ = []
    setattr(module, "_native_optimizer_module", native)
    get_install_context(tensor_type._frontend_backend).state["optimizer_frontend_native"] = MappingProxyType({
        "tensor_type": tensor_type,
        "base_init": native.Optimizer.__init__,
        "adam_add_param_group": native.Adam.add_param_group,
        "adamw_add_param_group": (native.AdamW.add_param_group
                                 if hasattr(native, "AdamW") else None),
        "algorithms": MappingProxyType({name: getattr(native, name).__init__
                                       for name in _ALGORITHM_INITIALIZERS if hasattr(native, name)}),
    })

    base = type("Optimizer", (native.Optimizer,), {
        "__module__": "torch.optim",
        "__init__": initialize_base,
    })
    setattr(module, "Optimizer", base)
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan", "Adagrad"):
        algorithm = getattr(native, name, None)
        if algorithm is None:
            continue
        # Native super() calls traverse this MRO through the frontend base,
        # whose state/closure adapters can be installed without changing the
        # original Optimizer or any native algorithm class dictionary.
        attributes = {
            "__module__": "torch.optim",
            "__init__": _ALGORITHM_INITIALIZERS[name],
        }
        if name == "Adam":
            attributes["add_param_group"] = add_adam_param_group
        if name == "AdamW":
            attributes.update(defaults=property(_adamw_defaults),
                              add_param_group=add_adamw_param_group)
        if name == "Adagrad":
            attributes["defaults"] = property(adagrad_defaults)
        setattr(module, name, type(name, (algorithm, base), attributes))
    for name in ("opt_grad", "LRScheduler", "LambdaLR"):
        if hasattr(native, name):
            setattr(module, name, getattr(native, name))
    setattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    return module
