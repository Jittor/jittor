"""Torch ``nn.modules`` namespace and global module registration hooks."""

import types

import jittor as jt

from .context import registry_for
from .types import _DEVICE_CTX_STACK, _set_meta_placeholder


def _meta_parameter_registration_device(module):
    """Return an active Accelerate-style meta registration target."""
    register = getattr(module, "register_parameter", None)
    function = getattr(register, "__func__", register)
    # Transformers 4.56 carries its own copy under
    # transformers.integrations.accelerate, so identify the bounded context by
    # its function/closure contract rather than one provider module name.
    if getattr(function, "__name__", None) != "register_empty_parameter":
        return None
    closure = getattr(function, "__closure__", None)
    freevars = getattr(getattr(function, "__code__", None), "co_freevars", ())
    if not closure:
        return None
    captured = dict(zip(freevars, (cell.cell_contents for cell in closure)))
    target = captured.get("device")
    return target if getattr(target, "type", None) == "meta" else None


def install_module_namespace(nn, registry=None):
    modules = registry_for(jt, registry).module_map
    modules_pkg = getattr(nn, "modules", None)
    if modules_pkg is None:
        try:
            from jittor.nn import modules as modules_pkg
        except Exception:
            modules_pkg = None
    if modules_pkg is None:
        modules_pkg = types.ModuleType("torch.nn.modules")
    modules["torch.nn.modules"] = modules_pkg
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

    class _ModuleRegistrationHandle:
        def __init__(self, hooks, hook_id):
            self.hooks = hooks
            self.id = hook_id

        def remove(self):
            self.hooks.pop(self.id, None)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.remove()
            return False

    def register_module_module_registration_hook(hook):
        if not callable(hook):
            raise TypeError("module registration hook must be callable")
        hook_id = max(registration_hooks, default=-1) + 1
        registration_hooks[hook_id] = hook
        return _ModuleRegistrationHandle(registration_hooks, hook_id)

    if not getattr(nn.Module.__setattr__, "_torch_module_registration_hooks", False):
        original_module_setattr = nn.Module.__setattr__

        def module_setattr(self, name, value):
            if isinstance(value, nn.Module):
                for hook in tuple(registration_hooks.values()):
                    result = hook(self, name, value)
                    if result is not None:
                        value = result
            elif isinstance(value, jt.Var) and _DEVICE_CTX_STACK:
                # Native Jittor initializers do not pass through the wrapped
                # torch factories. Mark Vars as they are installed on a module
                # so parameters created under `with torch.device("meta")`
                # retain that identity after the context exits.
                if not (
                    getattr(value, "_jittor_torch_force_cpu", False)
                    or getattr(value, "_jittor_torch_force_cuda", False)
                ):
                    _set_meta_placeholder(value)
            value_attrs = value.__dict__ if isinstance(value, jt.Var) else {}
            buffer_names = self.__dict__.get("_buffer_names", ())
            meta_parameter_candidate = (
                isinstance(value, jt.Var)
                and not name.startswith("_")
                and name not in buffer_names
                and value_attrs.get("is_buffer") is not True
                and value_attrs.get("persistent") is not False
                and (
                    value_attrs.get("_is_torch_parameter") is True
                    or value_attrs.get("_jt_plain_tensor") is not True
                )
            )
            if (meta_parameter_candidate
                    and _meta_parameter_registration_device(self) is not None):
                # Accelerate normally rewraps the result of Parameter.to(meta)
                # with type(param). A compatibility Parameter is a marked Var,
                # whose native constructor cannot accept PyTorch Parameter
                # kwargs. Register the same semantic parameter as the bounded
                # meta placeholder directly instead.
                _set_meta_placeholder(value)
            return original_module_setattr(self, name, value)

        module_setattr._torch_module_registration_hooks = True
        nn.Module.__setattr__ = module_setattr

    module_mod._global_module_registration_hooks = registration_hooks
    module_mod.register_module_module_registration_hook = register_module_module_registration_hook
    module_mod._IncompatibleKeys = getattr(
        module_mod,
        "_IncompatibleKeys",
        type(
            "_IncompatibleKeys",
            (tuple,),
            {
                "__new__": lambda cls, missing_keys, unexpected_keys: tuple.__new__(
                    cls, (missing_keys, unexpected_keys)
                ),
                "missing_keys": property(lambda self: self[0]),
                "unexpected_keys": property(lambda self: self[1]),
            },
        ),
    )
    modules_pkg.Module = nn.Module
    modules_pkg.module = module_mod
    for class_name in dir(nn):
        if class_name and class_name[0].isupper() and not hasattr(modules_pkg, class_name):
            try:
                setattr(modules_pkg, class_name, getattr(nn, class_name))
            except Exception:
                pass

    container_mod = modules.get("torch.nn.modules.container")
    if container_mod is None:
        container_mod = types.ModuleType("torch.nn.modules.container")
        modules["torch.nn.modules.container"] = container_mod
    for class_name in ("Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict"):
        if hasattr(nn, class_name):
            setattr(container_mod, class_name, getattr(nn, class_name))
    modules_pkg.container = container_mod
    return modules_pkg
