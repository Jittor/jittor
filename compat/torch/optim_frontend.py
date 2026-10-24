"""Independent optimizer types reusing native update implementations."""

from types import ModuleType, MappingProxyType
from jittor.optim.base import initialize_optimizer_storage, register_frontend_optimizer_root
from jittor.optim.algorithms.adam import initialize_adam_storage, add_adam_param_group
from collections.abc import Mapping
from .frontend import tensor_frontend
from .context import get_install_context
import jittor as jt


def initialize_base(self, params, defaults):
    if not isinstance(defaults, Mapping):
        raise TypeError("optimizer defaults must be a mapping")
    defaults = dict(defaults)
    lr = defaults.get("lr", 0.0)
    state = get_install_context(jt).state["optimizer_frontend_native"]
    tensor_type = state["tensor_type"]
    if jt.core._is_tensor_carrier(params):
        raise TypeError("optimizer params must be an iterable of tensors or parameter groups")
    params = list(params)
    for group in params:
        if isinstance(group, Mapping):
            if not isinstance(group, dict):
                raise TypeError('optimizer parameter group must be an actual dict')
            group["params"] = ([group["params"]]
                               if jt.core._is_tensor_carrier(group["params"])
                               else list(group["params"]))
    with tensor_frontend(tensor_type):
        initialize_optimizer_storage(self, params, lr)
    self.defaults = defaults
    for group in self.param_groups:
        for key, value in defaults.items():
            group.setdefault(key, value)


def _initialize_algorithm(name, instance, args, kwargs):
    state = get_install_context(jt).state["optimizer_frontend_native"]
    with tensor_frontend(state["tensor_type"]):
        state["algorithms"][name](instance, *args, **kwargs)


def initialize_sgd(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False,
                   *, foreach=None, fused=None):
    if foreach and fused:
        raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
    result = _initialize_algorithm("SGD", self,
        (params, lr, momentum, weight_decay, dampening, nesterov),
        {"torch_math": True, "fused": fused})
    self.foreach = foreach
    for group in self.param_groups:
        group.setdefault("foreach", foreach)
        group.setdefault("fused", fused)
    return result


def initialize_adam(self, *args, **kwargs):
    return _initialize_algorithm("Adam", self, args, kwargs)


def initialize_adamw(self, *args, **kwargs):
    return _initialize_algorithm("AdamW", self, args, kwargs)


def initialize_rmsprop(self, *args, **kwargs):
    return _initialize_algorithm("RMSprop", self, args, kwargs)


def initialize_adan(self, *args, **kwargs):
    return _initialize_algorithm("Adan", self, args, kwargs)


def initialize_adagrad(self, *args, **kwargs):
    raise NotImplementedError("torch.optim.Adagrad has no native Jittor update implementation")


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

    # Canonical owner module globals are NORMAL module namespaces. These are
    # literal class definitions; no cloning original globals/class dictionaries,
    # posthoc __module__/__bases__ edits, or synthetic MRO reports.
    owners = {}
    for suffix in ('optimizer', 'adam', 'adamw'):
        owners[suffix] = ModuleType('torch.optim.' + suffix)
        owners[suffix].__package__ = 'torch.optim'
    owners['optimizer']._initialize_base = initialize_base
    exec(_BASE_DECLARATION, owners['optimizer'].__dict__)
    base = owners['optimizer'].Optimizer
    for name in _SHARED_BASE_METHODS:
        setattr(base, name, getattr(native.Optimizer, name))
    base._execution_param_groups = _public_execution_param_groups
    register_frontend_optimizer_root(base)
    owners['adam'].Optimizer = base
    owners['adam']._initialize_adam_state = _initialize_public_adam_state
    owners['adam']._finish_adam_state = _finish_public_adam_state
    owners['adam']._add_adam_param_group = _public_add_adam_param_group
    exec(_ADAM_DECLARATION, owners['adam'].__dict__)
    owners['adamw'].Adam = owners['adam'].Adam
    exec(_ADAMW_DECLARATION, owners['adamw'].__dict__)
    module.Optimizer = base
    module.Adam = owners['adam'].Adam
    module.AdamW = owners['adamw'].AdamW
    module._canonical_optimizer_owners = MappingProxyType(owners)
    # Unmigrated algorithms KEEP their real native implementation and support.
    # This explicit legacy bridge is not advertised as MRO-equal to PyTorch.
    legacy = type('_LegacyNativeOptimizer', (base, native.Optimizer), {
        '__init__': _initialize_legacy_base})
    for name in ('SGD', 'RMSprop', 'Adan'):
        algorithm = getattr(native, name, None)
        if algorithm is not None:
            setattr(module, name, type(name, (algorithm, legacy), {
                '__module__': 'torch.optim', '__init__': _ALGORITHM_INITIALIZERS[name]}))
    # Libraries inspect optimizer type lists at import time. A missing native
    # algorithm remains explicitly unsupported when construction is requested.
    setattr(module, "Adagrad", type("Adagrad", (base,), {
        "__module__": "torch.optim", "__init__": initialize_adagrad,
    }))
    for name in ("opt_grad", "LRScheduler", "LambdaLR"):
        if hasattr(native, name):
            setattr(module, name, getattr(native, name))
    setattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    return module


_SHARED_BASE_METHODS = (
    'add_param_group', '_advance_step_count', 'set_input_into_param_group',
    'clip_grad_norm', 'state_dict', 'load_state_dict', 'zero_grad', 'backward',
    'pre_step', 'post_step', 'step', '_build_grad_map', 'find_grad',
)


def _initialize_public_adam_state(self, eps, betas, weight_decay):
    tensor_type = get_install_context(jt).state['optimizer_frontend_native']['tensor_type']
    with tensor_frontend(tensor_type):
        initialize_adam_storage(self, eps, betas, weight_decay)
    self.defaults.update(eps=eps, betas=betas, weight_decay=weight_decay)


def _finish_public_adam_state(self, foreach, *, fused, capturable,
                              differentiable, amsgrad, maximize, decoupled_weight_decay):
    from . import optimizer_api as api
    options = dict(foreach=foreach, fused=fused, capturable=capturable,
        differentiable=differentiable, amsgrad=amsgrad, maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay)
    for flag in ('capturable', 'differentiable', 'amsgrad', 'maximize', 'decoupled_weight_decay'):
        if type(options[flag]) is not bool:
            raise TypeError('Adam ' + flag + ' must be bool')
    kind = api._torch_optimizer_kind(self)
    api._adam_state_decoupled_policy(kind, options)
    if kind == 'adam' and fused is True:
        raise NotImplementedError('Adam fused=True update is not admitted')
    # Preserve every supplied False value. Never pop/drop the keyword or change
    # user config to make a supported eager branch look like capture support.
    self.defaults.update(options)
    for group in self.param_groups:
        for key, value in self.defaults.items():
            group.setdefault(key, value)
        api._adam_state_decoupled_policy(kind, group)
    api._persistent_adam_owner(self).validate_groups()


def _initialize_legacy_base(self, params, lr, param_sync_iter=10000):
    from .optimizer_api import _register_initialized_optimizer
    # Native subclasses retain their read-only defaults descriptor. Reuse the
    # real native initializer, not the independent root's mutable defaults API.
    state = get_install_context(jt).state['optimizer_frontend_native']
    state['base_init'](self, params, lr, param_sync_iter)
    _register_initialized_optimizer(self)


_BASE_DECLARATION = '\nclass Optimizer:\n    def __init__(self, params, defaults):\n        _initialize_base(self, params, defaults)\n'
_ADAM_DECLARATION = "\nclass Adam(Optimizer):\n    _torch_lr_default = True\n    def __init__(self, params, lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=0, *, foreach=None, fused=None, capturable=False, differentiable=False, amsgrad=False, maximize=False, decoupled_weight_decay=False):\n        super().__init__(params, {'lr': lr, 'eps': eps, 'betas': betas, 'weight_decay': weight_decay})\n        self.fused = fused\n        _initialize_adam_state(self, eps, betas, weight_decay)\n        _finish_adam_state(self, foreach, fused=fused, capturable=capturable,\n            differentiable=differentiable, amsgrad=amsgrad, maximize=maximize,\n            decoupled_weight_decay=decoupled_weight_decay)\n    add_param_group = _add_adam_param_group\n"
_ADAMW_DECLARATION = '\nclass AdamW(Adam):\n    _torch_lr_default = True\n    def __init__(self, params, lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.01, fused=None, *, foreach=None, capturable=False, differentiable=False, amsgrad=False, maximize=False):\n        super().__init__(params, lr, eps, betas, weight_decay, foreach=foreach,\n            fused=fused, capturable=capturable, differentiable=differentiable,\n            amsgrad=amsgrad, maximize=maximize, decoupled_weight_decay=True)\n'


def _public_execution_param_groups(self, *, include=None):
    from . import optimizer_api as api
    if api._torch_optimizer_kind(self) in ('adam', 'adamw'):
        return api._persistent_adam_owner(self).execution_groups(include=include)
    # Unmigrated algorithms retain their original native dictionaries/support.
    return self.param_groups if include is None else [include]


def _public_add_adam_param_group(self, group):
    from . import optimizer_api as api
    if not isinstance(group, dict):
        raise TypeError('optimizer parameter group must be an actual dict')
    if jt.core._is_tensor_carrier(group['params']):
        group['params'] = [group['params']]
    for key, value in self.defaults.items():
        group.setdefault(key, value)
    api._adam_state_decoupled_policy(api._torch_optimizer_kind(self), group)
    api._persistent_adam_owner(self).policy.validate_group(group)
    tensor_type = get_install_context(jt).state['optimizer_frontend_native']['tensor_type']
    with tensor_frontend(tensor_type):
        add_adam_param_group(self, group)
