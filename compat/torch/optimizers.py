"""Torch optimizer behavior layered over Jittor optimizers."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from collections.abc import Mapping

import jittor as jt
import numpy as np

from .context import registry_for
from .types import _dtype_to_str
from ..diagnostics import EXPECTED, swallowed
from .. import fsdp_hooks as _fsdp_hooks
from .. import optimizer_kinds as _optimizer_kinds
from .tensor_state import get_tensor_state


from .optimizer_api import (
    adam_step, adamw_step, _STEP_APIS, _INITIALIZER_APIS,
    _state_getter, _lbfgs_type,
    LBFGS,
    _OptState,
    _ParamState,
    _advance_ready_param_steps,
    _advance_trainable_param_steps,
    _backward_with_step_marker,
    _init,
    _load_fsdp2_for_optimizer,
    _load_state_dict_torch,
    _lsd,
    _optimizer_has_ready_grads,
    _optimizer_maybe_has_fsdp_params,
    _state_dict_torch,
    _torch_optimizer_kind,
    _torch_param_steps,
    _zero_grad_compat,
)
from .context import get_install_context
from .fidelity import Fidelity, register_api_bindings
from types import MappingProxyType

def _install_optimizers(g, registry=None):
    """Register optimizer instances weakly on construction and mirror lr into
    each param_group. This makes the
    `loss.backward()` bridge (Var.backward) and torch-style LR schedulers work
    even when using `import jittor as torch` directly (no torch_shim wrapper)."""
    _registry = registry_for(g, registry)
    _modules = _registry.module_map
    from jittor import optim as _optim
    from jittor.optim.algorithms.adam import adam_update
    if g is not _registry.native_backend:
        from .optim_frontend import make_optimizer_frontend
        existing = vars(g).get("optim")
        if (existing is None or
                vars(existing).get("_native_optimizer_module") is not _optim):
            existing = make_optimizer_frontend(_optim, g.Var)
            g.optim = existing
        _optim = existing
    Base = getattr(_optim, "Optimizer", None)
    if Base is None:
        raise RuntimeError("jittor.optim has no Optimizer owner")
    if getattr(Base, "_torch_compat_wrapped", False):
        _modules.setdefault("torch.optim", _optim)
        _optim.__path__ = getattr(_optim, "__path__", [])
        import types as _types_optim
        _optim_sub = _modules.get("torch.optim.optimizer")
        if _optim_sub is None:
            _optim_sub = _types_optim.ModuleType("torch.optim.optimizer")
            _modules["torch.optim.optimizer"] = _optim_sub
        _optim_sub.Optimizer = Base
        _optim_sub.ParamsT = object
        return
    import weakref as _weakref
    _native_steps = {}
    _native_initializers = {}
    _orig_init = Base.__init__
    Base.__init__ = _init
    # torch-compatible Optimizer.state: a mapping keyed by the parameter object,
    # each value a dict {"exp_avg","exp_avg_sq","step"} backed by jittor's
    # positional per-group state lists pg["m"] (exp_avg) / pg["values"] (exp_avg_sq).
    # 3DGS densification does surgery on this (read state.get(p), mutate exp_avg/
    # exp_avg_sq via mask/cat, del old key, set new key after replacing the param).
    if not hasattr(Base, "_torch_state_installed"):
        Base._OptState = _OptState
        Base.state = property(_state_getter)
        Base._torch_state_installed = True
    if not getattr(Base, "_torch_state_dict_wrapped", False):
        _native_load_state_dict = Base.load_state_dict
        Base.state_dict = _state_dict_torch
        Base.load_state_dict = _load_state_dict_torch
        Base._torch_state_dict_wrapped = True
    # torch's Optimizer.zero_grad accepts set_to_none=; jittor's rejects the kwarg.
    if not getattr(Base, "_torch_zero_grad_wrapped", False):
        _orig_zero = Base.zero_grad
        Base.zero_grad = _zero_grad_compat
        Base._torch_zero_grad_wrapped = True
    # Native Optimizer.backward() advances n_step; tensor.backward() below does
    # not. Record which spelling produced the ready gradient so a subsequent
    # torch-style step() advances the counter exactly once in either case.
    if not getattr(Base, "_torch_backward_step_marker", False):
        _orig_backward = Base.backward
        Base.backward = _backward_with_step_marker
        Base._torch_backward_step_marker = True
    # torch's Adam/AdamW default lr=1e-3 (jittor makes lr positional-required).
    # 3DGS builds the exposure optimizer as torch.optim.Adam([self._exposure]).
    for _cls_name in ("Adam", "AdamW", "RMSprop", "Adan"):
        _cls = getattr(_optim, _cls_name, None)
        if _cls is None or getattr(_cls, "_torch_lr_default", False):
            continue
        _ci = _cls.__init__
        _native_initializers[_cls_name] = _ci
        _cls.__init__ = _INITIALIZER_APIS[_cls_name]
        _cls._torch_lr_default = True
    Adam = getattr(_optim, "Adam", None)
    if Adam is not None and not getattr(Adam, "_torch_adam_step", False):
        Adam.step = adam_step
        Adam._torch_adam_step = True
    AdamW = getattr(_optim, "AdamW", None)
    if AdamW is not None and not getattr(AdamW, "_torch_adamw_step", False):
        AdamW.step = adamw_step
        AdamW._torch_adamw_step = True
    for _cls_name, _native_kind in (("SGD", "sgd"), ("RMSprop", "rmsprop"), ("Adan", "adan")):
        _cls = getattr(_optim, _cls_name, None)
        if _cls is not None and not getattr(_cls, "_torch_closure_step", False):
            _native_steps[_native_kind] = _cls.step
            _cls.step = _STEP_APIS[_native_kind]
            _cls._torch_closure_step = True
    Base._torch_compat_wrapped = True
    if not hasattr(_optim, "LBFGS"):
        _optim.LBFGS = _lbfgs_type(Base)
    if g is not _registry.native_backend:
        _optim.__all__ = sorted(name for name in vars(_optim)
                               if not name.startswith("_"))

    import types as _types_optim
    _optim_mod = _modules.get("torch.optim")
    if _optim_mod is None:
        _modules["torch.optim"] = _optim
        _optim_mod = _optim
    _optim_mod.__path__ = getattr(_optim_mod, "__path__", [])
    if not hasattr(_optim_mod, "Optimizer"):
        _optim_mod.Optimizer = Base
    if not hasattr(_optim_mod, "LBFGS"):
        _optim_mod.LBFGS = _optim.LBFGS
    _optim_sub = _modules.get("torch.optim.optimizer")
    if _optim_sub is None:
        _optim_sub = _types_optim.ModuleType("torch.optim.optimizer")
        _modules["torch.optim.optimizer"] = _optim_sub
    _optim_sub.Optimizer = Base
    _optim_sub.ParamsT = object

    # jittor's load_state_dict runs a dfs that calls .stop_grad() on every Var
    # it meets -- including params nested under param_groups -- freezing all
    # trainable params (accelerate round-trips state_dict on wrap). Guard it.
    _orig_lsd = getattr(Base, "load_state_dict", None)
    if _orig_lsd is not None:
        Base.load_state_dict = _lsd


    get_install_context(g).state["optimizer_native_api"] = MappingProxyType({
        "steps": MappingProxyType(_native_steps),
        "initializers": MappingProxyType(_native_initializers),
        '_native_load_state_dict': locals().get('_native_load_state_dict'),
        '_orig_backward': locals().get('_orig_backward'),
        '_orig_init': locals().get('_orig_init'),
        '_orig_lsd': locals().get('_orig_lsd'),
        '_orig_zero': locals().get('_orig_zero'),
    })
    register_api_bindings(_optim, "torch.optim",
        ("Optimizer", "SGD", "Adam", "AdamW", "RMSprop", "Adan"),
        Fidelity.APPROXIMATE,
        "Installation-owned types reuse native optimizer mathematics; supported "
        "group options, closure behavior and device capabilities are restricted")
    register_api_bindings(Base, "torch.optim.Optimizer",
        ("__init__", "state", "state_dict", "load_state_dict", "zero_grad", "backward"),
        Fidelity.APPROXIMATE,
        "State and gradient adapters over native parameter groups; only known "
        "optimizer state layouts and supported restore formats are handled")
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan"):
        algorithm = getattr(_optim, name, None)
        if algorithm is not None:
            register_api_bindings(algorithm, "torch.optim." + name,
                ("__init__", "step"), Fidelity.APPROXIMATE,
                "Stable native-update adapter preserving per-parameter steps, "
                "gradient retention and the registered FSDP provider boundary")
    register_api_bindings(_optim.LBFGS, "torch.optim.LBFGS", ("step",),
        Fidelity.UNIMPLEMENTED, "LBFGS updates explicitly raise NotImplementedError")


def install_module_keys(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    optim = g.optim
    for suffix, class_name, fallback in (
        ("sgd", "SGD", None),
        ("adam", "Adam", None),
        ("adamw", "AdamW", "Adam"),
        ("rmsprop", "RMSprop", None),
    ):
        optim_module = registry.ensure("torch.optim." + suffix)
        value = getattr(optim, class_name, None)
        if value is None and fallback is not None:
            value = getattr(optim, fallback, None)
        if value is not None:
            setattr(optim_module, class_name, value)
        setattr(optim, suffix, optim_module)
