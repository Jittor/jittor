"""Torch ``nn.modules`` namespace and global module registration hooks."""

import types

import jittor as jt

from .context import registry_for


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
