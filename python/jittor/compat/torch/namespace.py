"""An explicit module boundary for the eventual independent Torch package.

The current compatibility install still publishes Jittor as ``torch`` for
backwards compatibility.  ``TorchNamespace`` provides the next boundary:
it has a distinct module identity while delegating the native implementation
to an explicitly supplied Jittor owner.  It is intentionally opt-in until
the installer families have all stopped assuming that their owner is the
native Jittor module.
"""

from __future__ import annotations

import types


class TorchNamespace(types.ModuleType):
    """Module-shaped view over one Jittor compatibility owner.

    Public reads and writes are delegated to ``owner`` so existing installers
    can be migrated family by family.  Private bookkeeping stays on this
    module, which prevents a future independent ``torch`` package from
    leaking its install markers into the native Jittor namespace.
    """

    def __init__(self, owner):
        if owner is None:
            raise TypeError("TorchNamespace requires an owner module")
        super().__init__("torch")
        object.__setattr__(self, "_torch_owner", owner)
        self.__package__ = "torch"
        self.__path__ = []

    @property
    def owner(self):
        return object.__getattribute__(self, "_torch_owner")

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in {"__name__", "__package__", "__path__"}:
            return super().__setattr__(name, value)
        setattr(self.owner, name, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.owner)))


def independent_torch_namespace(owner):
    """Return a distinct, module-shaped Torch namespace for ``owner``."""

    return TorchNamespace(owner)


def namespace_owner(module):
    """Return the Jittor owner for an independent namespace, if any."""

    if isinstance(module, TorchNamespace):
        return module.owner
    return None


def bind_published_namespace(namespace, published, transaction=None):
    """Bind published ``torch.*`` children to an independent root.

    Installers publish children while the compatibility owner is still the
    ``torch`` root.  Once that root is replaced by :class:`TorchNamespace`,
    parent attributes must be rebound to the independent module tree as well.
    ``transaction`` records every attribute mutation so a failed activation
    restores the owner's original bindings.
    """
    if not isinstance(namespace, TorchNamespace):
        raise TypeError("namespace must be a TorchNamespace")
    modules = {
        name: module for name, module in published.items()
        if name.startswith("torch.") and module is not None
    }
    parents = {"torch": namespace}
    for name in sorted(modules, key=lambda item: (item.count("."), item)):
        # The registry includes its root entry as well as children.  The root
        # is published by the activation transaction; only dotted entries
        # need a parent attribute binding here.
        if name == "torch":
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = parents.get(parent_name)
        if parent is None:
            # A published child without its published parent cannot be
            # represented by an independent namespace.  Silently skipping it
            # leaves registry/sys.modules claiming success while attribute
            # imports resolve through the old owner (or fail later).
            raise RuntimeError(
                "cannot bind published module %r: parent %r is not published"
                % (name, parent_name)
            )
        had_attr = hasattr(parent, attr)
        old = getattr(parent, attr, None)
        if transaction is not None:
            def undo(parent=parent, attr=attr, had_attr=had_attr, old=old):
                if had_attr:
                    object.__setattr__(parent, attr, old)
                else:
                    try:
                        object.__delattr__(parent, attr)
                    except AttributeError:
                        pass
            transaction.record_undo(undo)
        object.__setattr__(parent, attr, modules[name])
        parents[name] = modules[name]
    return namespace


__all__ = [
    "TorchNamespace", "independent_torch_namespace", "namespace_owner",
    "bind_published_namespace",
]
