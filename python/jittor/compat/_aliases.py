"""Fixed, same-object compatibility aliases for migrated Jittor modules."""

from __future__ import absolute_import

import importlib
import importlib.abc
import importlib.util
import os
import sys


ALIASES = {
    "jittor.attention": "jittor.nn.attention",
    "jittor.contrib": "jittor.compat.contrib",
    "jittor.gradfunctional": "jittor.autograd",
    "jittor.gradfunctional.functional": "jittor.autograd.functional",
    "jittor.other": "jittor.nn.backends",
    "jittor.other.code_softmax": "jittor.nn.backends.softmax_cuda",
    "jittor.lr_scheduler": "jittor.optim.legacy_schedulers",
    "jittor.nn.sparse": "jittor.sparse.convolution",
    "jittor.weightnorm": "jittor.nn.utils.weight_norm",
    "jittor.torch_compat": "jittor.compat.torch",
    "jittor.torch_compat.context": "jittor.compat.torch.context",
    "jittor.torch_compat.functional": "jittor.compat.torch.functional",
    "jittor.torch_compat.grad": "jittor.compat.torch.grad",
    "jittor.torch_compat.lr_scheduler": "jittor.compat.torch.lr_scheduler",
    "jittor.torch_compat.nested": "jittor.compat.torch.nested",
    "jittor.torch_compat.optimizers": "jittor.compat.torch.optimizers",
    "jittor.torch_compat.serialization": "jittor.compat.torch.serialization",
    "jittor.torch_compat.types": "jittor.compat.torch.types",
    "jittor.torch_fsdp2_compat": "jittor.compat.fsdp2",
    "jittor.torch_fsdp2_compat.api": "jittor.compat.fsdp2.api",
    "jittor.torch_fsdp2_compat.common": "jittor.compat.fsdp2.common",
    "jittor.torch_fsdp2_compat.compat_types": "jittor.compat.fsdp2.compat_types",
    "jittor.torch_fsdp2_compat.config": "jittor.compat.fsdp2.config",
    "jittor.torch_fsdp2_compat.dtensor": "jittor.compat.fsdp2.dtensor",
    "jittor.torch_fsdp2_compat.grad_sync": "jittor.compat.fsdp2.grad_sync",
    "jittor.torch_fsdp2_compat.installer": "jittor.compat.fsdp2.installer",
    "jittor.torch_fsdp2_compat.optimizer": "jittor.compat.fsdp2.optimizer",
    "jittor.torch_fsdp2_compat.shard": "jittor.compat.fsdp2.shard",
    "jittor.torch_shim": "jittor.compat.shim",
    "jittor.torch_shim.bootstrap": "jittor.compat.shim.bootstrap",
    "jittor.torch_shim.deploy": "jittor.compat.shim.deploy",
    "jittor.torch_shim.cpp_extension": "jittor.compat.shim.cpp_extension",
    "jittor.torch_shim.cpp_extension.torch_utils": "jittor.compat.shim.cpp_extension.torch_utils",
    "jittor.torch_shim.torch_utils": "jittor.compat.shim.cpp_extension.torch_utils",
    "jittor.torch_shim.flashattn_jittor": "jittor.compat.shim.backends.flash_attention",
    "jittor.torch_shim.flashattn": "jittor.compat.shim.backends.flash_attention",
    "jittor.torch_shim.flash_attention": "jittor.compat.shim.backends.flash_attention",
    "jittor.torch_shim.readonly_extensions": "jittor.compat.shim.extensions.readonly",
    "jittor.torch_shim.readonly": "jittor.compat.shim.extensions.readonly",
    "jittor.compat.shim.torch_utils": "jittor.compat.shim.cpp_extension.torch_utils",
    "jittor.compat.shim.flashattn_jittor": "jittor.compat.shim.backends.flash_attention",
    "jittor.compat.shim.flashattn": "jittor.compat.shim.backends.flash_attention",
    "jittor.compat.shim.readonly_extensions": "jittor.compat.shim.extensions.readonly",
    "jittor.compat.shim.readonly": "jittor.compat.shim.extensions.readonly",
    "jittor.triton_shim": "jittor.compat.triton",
    "jittor.triton_shim.backend": "jittor.compat.triton.backend",
    "jittor.triton_shim.deploy": "jittor.compat.triton.deploy",
    "jittor.triton_shim.language": "jittor.compat.triton.language",
    "jittor.triton_shim.launch": "jittor.compat.triton.launch",
    "jittor.depthwise_conv": "jittor.nn.modules.depthwise",
}

_PACKAGE_TARGETS = frozenset(
    (
        "jittor.compat.torch",
        "jittor.compat.fsdp2",
        "jittor.compat.shim",
        "jittor.compat.shim.cpp_extension",
        "jittor.compat.triton",
        "jittor.autograd",
        "jittor.nn.backends",
        "jittor.sparse",
    )
)
_LAZY_PARENT_BINDINGS = frozenset(
    alias for alias in ALIASES if alias.startswith("jittor.torch_fsdp2_compat.")
)
_TORCH_PARENT_BINDING_EXCEPTIONS = frozenset(("torch.distributed._composable.fsdp.fully_shard",))


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias, canonical):
        self.alias = alias
        self.canonical = canonical
        self.metadata = None

    def create_module(self, spec):
        module = importlib.import_module(self.canonical)
        self.metadata = (
            module.__name__,
            module.__package__,
            module.__loader__,
            module.__spec__,
        )
        return module

    def exec_module(self, module):
        module.__name__, module.__package__, module.__loader__, module.__spec__ = self.metadata
        _publish_alias(self.alias, module)
        if self.alias == "jittor.torch_compat":
            root_module = sys.modules.get("jittor")
            if root_module is not None:
                module.install(root_module)


class _AliasFinder(importlib.abc.MetaPathFinder):
    _jittor_compat_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        canonical = ALIASES.get(fullname)
        if canonical is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, canonical),
            is_package=canonical in _PACKAGE_TARGETS,
        )


_FINDER = _AliasFinder()


def _is_deployed_torch_placeholder(module):
    source = getattr(module, "__file__", None)
    if not source:
        return False
    path = os.path.realpath(os.fspath(source))
    return bool(
        getattr(module, "_jittor_torch_shim_placeholder", False)
        and getattr(module, "__name__", "") == "torch"
        and os.path.basename(path) == "__init__.py"
        and os.path.basename(os.path.dirname(path)) == "torch"
    )


def _torch_namespace():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "torch" or name.startswith("torch.")
    }


def torch_namespace_owned(root_module):
    """Return whether the complete Torch graph is still owned by ``root_module``."""

    namespace = {name: module for name, module in _torch_namespace().items()}
    torch_root = namespace.get("torch")
    if torch_root is not root_module:
        from .torch.namespace import namespace_owner
        if namespace_owner(torch_root) is not root_module:
            return False
    context = getattr(root_module, "_torch_compat_install_context", None)
    registry = getattr(context, "registry", None)
    published = getattr(registry, "_published", None)
    if not published:
        return False
    owned = {
        name: module
        for name, module in published.items()
        if name == "torch" or name.startswith("torch.")
    }
    if not (
        namespace.keys() == owned.keys()
        and all(namespace[name] is module for name, module in owned.items())
    ):
        return False
    for name, module in owned.items():
        if "." not in name or name in _TORCH_PARENT_BINDING_EXCEPTIONS:
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = owned.get(parent_name)
        if parent is not None and getattr(parent, attr, None) is not module:
            return False
    return True


def torch_namespace_claimable(root_module):
    namespace = _torch_namespace()
    current = namespace.get("torch")
    children = tuple(name for name in namespace if name.startswith("torch."))
    if current is None:
        return not children
    from .torch.namespace import namespace_owner
    if current is root_module or namespace_owner(current) is root_module:
        context = getattr(root_module, "_torch_compat_install_context", None)
        registry = getattr(context, "registry", None)
        if getattr(registry, "_published", None):
            return torch_namespace_owned(root_module)
        return not children
    return _is_deployed_torch_placeholder(current) and not children


def torch_compat_requested(root_module, preflight=None):
    """Return whether this process explicitly selected the Torch API mode."""

    if bool(getattr(preflight, "active", False)):
        return True
    current = _torch_namespace().get("torch")
    return _is_deployed_torch_placeholder(current) or torch_namespace_owned(root_module)


def _bind_parent(alias, module):
    if "." not in alias:
        return
    parent_name, attr = alias.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr, module)


def _publish_alias(alias, module, bind_parent=True):
    current = sys.modules.get(alias)
    if current is not None and current is not module:
        raise RuntimeError("module alias %r already published with a different object" % alias)
    sys.modules[alias] = module
    if bind_parent:
        _bind_parent(alias, module)
    return module


def publish_loaded_aliases(root_module=None):
    for alias, canonical in ALIASES.items():
        if alias in _LAZY_PARENT_BINDINGS:
            continue
        module = sys.modules.get(canonical)
        if module is None:
            continue
        _publish_alias(alias, module)
    if root_module is not None:
        for attr, canonical in (
            ("attention", "jittor.nn.attention"),
            ("lr_scheduler", "jittor.optim.legacy_schedulers"),
            ("torch_compat", "jittor.compat.torch"),
            ("torch_fsdp2_compat", "jittor.compat.fsdp2"),
            ("torch_shim", "jittor.compat.shim"),
            ("triton_shim", "jittor.compat.triton"),
            ("depthwise_conv", "jittor.nn.modules.depthwise"),
        ):
            module = sys.modules.get(canonical)
            if module is not None:
                setattr(root_module, attr, module)


def install_aliases(root_module=None):
    if not any(getattr(finder, "_jittor_compat_alias_finder", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)
    publish_loaded_aliases(root_module)
    return dict(ALIASES)


def import_alias(alias):
    canonical = ALIASES[alias]
    module = importlib.import_module(canonical)
    return _publish_alias(alias, module)
