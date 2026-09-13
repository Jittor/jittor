"""Torch ``nn.modules`` namespace and global module registration hooks."""

import types
import weakref
from typing import Any, cast
from collections import namedtuple

import jittor as jt

from .context import registry_for, get_install_context
from .api_delegates import bind_delegates
from .fidelity import Fidelity, register_fidelity
from ..transaction import set_attr
from ..diagnostics import EXPECTED, swallowed
from .types import _DEVICE_CTX_STACK, _set_meta_placeholder


class _ModuleRegistrationHandle:
    def __init__(self, hooks, hook_id):
        self.hooks, self.id = hooks, hook_id
        self._hook = hooks[hook_id]

    def remove(self):
        if self.hooks.get(self.id) is self._hook:
            self.hooks.pop(self.id, None)
        self._hook = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


class _IncompatibleKeys(namedtuple("IncompatibleKeys", "missing_keys unexpected_keys")):
    __slots__ = ()


def register_module_module_registration_hook(hook):
    if not callable(hook):
        raise TypeError("module registration hook must be callable")
    state = get_install_context(jt).state["nn_module_registration"]
    hooks = state["hooks"]
    hook_id = max(hooks, default=-1) + 1
    hooks[hook_id] = hook
    return _ModuleRegistrationHandle(hooks, hook_id)


def module_setattr(self, name, value):
    reference = getattr(type(self), "_torch_registration_context", None)
    context = reference() if reference is not None else None
    if context is None:
        raise RuntimeError("Module registration owner is no longer active")
    state = context.state["nn_module_registration"]
    if isinstance(value, state["module_type"]):
        for hook in tuple(state["hooks"].values()):
            replacement = hook(self, name, value)
            if replacement is not None:
                value = replacement
    elif isinstance(value, jt.Var):
        parameter_type = context.state.get("Parameter")
        register_parameter = getattr(type(self), "register_parameter", None)
        if (parameter_type is not None and isinstance(value, parameter_type)
                and callable(register_parameter)
                and not getattr(register_parameter,
                                "_jittor_torch_native_registration", False)):
            return register_parameter(self, name, value)
        if (_DEVICE_CTX_STACK
                and (getattr(value, "_jittor_torch_meta", False)
                     or getattr(value, "placement_backend", -1) < 0)):
            _set_meta_placeholder(value)
    return state["setattr"](self, name, value)


setattr(cast(Any, module_setattr), "_torch_module_registration_hooks", True)


def install_module_namespace(nn, registry=None):
    modules = registry_for(jt, registry).module_map
    modules_pkg = getattr(nn, "modules", None)
    if modules_pkg is None:
        try:
            from jittor.nn import modules as imported_modules_pkg
        except EXPECTED as exc:
            swallowed("torch/nn_modules.py install_module_namespace: from jittor.nn import modules as modules_pkg", exc)
            modules_pkg = imported_modules_pkg
    if modules_pkg is None:
        modules_pkg = types.ModuleType("torch.nn.modules")
    modules["torch.nn.modules"] = modules_pkg
    modules_pkg = cast(Any, modules_pkg)
    modules_pkg.__path__ = getattr(modules_pkg, "__path__", [])

    module_mod = modules.get("torch.nn.modules.module")
    if module_mod is None:
        module_mod = types.ModuleType("torch.nn.modules.module")
        modules["torch.nn.modules.module"] = module_mod
    module_mod.Module = nn.Module
    module_mod._EXTRA_STATE_KEY_SUFFIX = "_extra_state"
    module_mod._global_backward_hooks = getattr(module_mod, "_global_backward_hooks", {})
    module_mod._global_forward_hooks = getattr(module_mod, "_global_forward_hooks", {})
    module_mod._global_forward_pre_hooks = getattr(module_mod, "_global_forward_pre_hooks", {})

    registration_hooks = getattr(nn.Module, "_torch_global_module_registration_hooks", None)
    if registration_hooks is None:
        registration_hooks = {}
        nn.Module._torch_global_module_registration_hooks = registration_hooks

    context = get_install_context(registry_for(jt, registry).target_namespace)
    state = context.state.get("nn_module_registration")
    if state is None or state["module_type"] is not nn.Module:
        original = nn.Module.__setattr__
        if original is module_setattr:
            raise RuntimeError("Module registration hooks already belong to another context")
        bind_delegates(context, "nn_module_registration", {
            "module_type": nn.Module, "hooks": registration_hooks, "setattr": original,
        })
    set_attr(nn.Module, "_torch_registration_context", weakref.ref(context), context=context)
    set_attr(nn.Module, "__setattr__", module_setattr, context=context)

    module_mod._global_module_registration_hooks = registration_hooks
    module_mod.register_module_module_registration_hook = register_module_module_registration_hook
    module_mod._IncompatibleKeys = getattr(module_mod, "_IncompatibleKeys", _IncompatibleKeys)
    register_fidelity("torch.nn.modules.module.register_module_module_registration_hook",
                      register_module_module_registration_hook, Fidelity.APPROXIMATE,
                      "Ordered module replacement hooks; ownership follows the active frontend context")
    setattr(modules_pkg, "Module", nn.Module)
    setattr(modules_pkg, "module", module_mod)
    for class_name in dir(nn):
        if class_name and class_name[0].isupper() and not hasattr(modules_pkg, class_name):
            try:
                setattr(modules_pkg, class_name, getattr(nn, class_name))
            except (AttributeError, TypeError) as exc:
                swallowed("torch/nn_modules.py install_module_namespace: setattr(modules_pkg, class_name, getattr(nn, class_name))", exc)

    container_mod = modules.get("torch.nn.modules.container")
    if container_mod is None:
        container_mod = types.ModuleType("torch.nn.modules.container")
        modules["torch.nn.modules.container"] = container_mod
    for class_name in ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict"):
        if hasattr(nn, class_name):
            setattr(container_mod, class_name, getattr(nn, class_name))
    setattr(modules_pkg, "container", container_mod)
    return modules_pkg
