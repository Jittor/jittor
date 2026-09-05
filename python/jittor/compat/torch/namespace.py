"""An explicit module boundary for the eventual independent Torch package.

The current compatibility install still publishes Jittor as ``torch`` for
backwards compatibility.  ``TorchNamespace`` provides the next boundary:
it has a distinct module identity while delegating the native implementation
to an explicitly supplied Jittor owner.  It is intentionally opt-in until
the installer families have all stopped assuming that their owner is the
native Jittor module.
"""

from __future__ import annotations

import importlib.machinery
import types


class TorchNamespace(types.ModuleType):
    """Module-shaped view over one Jittor compatibility owner.

    Public reads and writes are delegated to ``owner`` so existing installers
    can be migrated family by family.  Private bookkeeping stays on this
    module, which prevents a future independent ``torch`` package from
    leaking its install markers into the native Jittor namespace.
    """

    _LOCAL_METADATA = frozenset({
        "__name__", "__loader__", "__package__", "__spec__", "__path__",
        "__file__", "__cached__", "__builtins__", "__doc__",
    })

    def __init__(self, owner):
        if owner is None:
            raise TypeError("TorchNamespace requires an owner module")
        super().__init__("torch")
        object.__setattr__(self, "_torch_owner", owner)
        # Import metadata belongs to this detached module.  In particular,
        # assigning ``__spec__`` through the public delegation path would
        # silently write it onto the Jittor owner and make the package look
        # importable only by accident through ``__getattr__``.
        object.__setattr__(self, "__package__", "torch")
        # A detached module is not created by the import machinery, so it
        # otherwise has ``__spec__ = None``.  That makes importlib treat the
        # published root as a broken module even though its children are
        # already registered in sys.modules.
        object.__setattr__(self, "__loader__", None)
        # The view owns a valid package spec without importing activation or
        # registry machinery.  This keeps the standalone package boundary
        # usable on a machine that has not loaded the compatibility installer.
        object.__setattr__(self, "__spec__", importlib.machinery.ModuleSpec(
            "torch", loader=None, is_package=True
        ))
        object.__setattr__(self, "__path__", [])

    @property
    def owner(self):
        return object.__getattribute__(self, "_torch_owner")

    def __getattr__(self, name):
        # Import metadata is owned by this detached module.  Once a caller
        # removes a local metadata field, do not resurrect the owner's value
        # through the public delegation path (e.g. ``owner.__file__``).
        if name in self._LOCAL_METADATA:
            raise AttributeError(name)
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self._LOCAL_METADATA:
            return super().__setattr__(name, value)
        setattr(self.owner, name, value)

    def __delattr__(self, name):
        """Keep public mutation symmetry with :meth:`__setattr__`.

        Installer code occasionally removes an optional public binding.  A
        detached namespace must remove that binding from its explicit owner;
        deleting it only from the view would leave the owner unexpectedly
        populated and make a later install observe stale state.
        """
        if name.startswith("_") or name in self._LOCAL_METADATA:
            return super().__delattr__(name)
        delattr(self.owner, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.owner)))


def independent_torch_namespace(owner):
    """Compatibility import for the standalone publication helper."""

    from .publication import independent_torch_namespace as publish
    return publish(owner)


def namespace_owner(module):
    """Compatibility import for the standalone publication helper."""

    from .publication import namespace_owner as owner_of
    return owner_of(module)


def bind_published_namespace(namespace, published, transaction=None):
    """Compatibility import for the standalone publication helper."""

    from .publication import bind_published_namespace as bind
    return bind(namespace, published, transaction=transaction)


__all__ = [
    "TorchNamespace", "independent_torch_namespace", "namespace_owner",
    "bind_published_namespace",
]
