"""Publish an independent ``torch`` module graph.

This module is deliberately separate from :mod:`namespace`: the latter only
implements a module-shaped view, while this file owns installer registry
mutation and parent binding.  The split lets a future ``jittor-torch``
distribution import the view without pulling in the native shim installer.
"""

from __future__ import annotations

from .namespace import TorchNamespace
from ..transaction import _MISSING, TransactionConflict


def independent_torch_namespace(owner):
    """Return a distinct, module-shaped Torch namespace for ``owner``."""

    return TorchNamespace(owner)


def namespace_owner(module):
    """Return the explicit native owner for an independent namespace."""

    if isinstance(module, TorchNamespace):
        return module.owner
    return None


def bind_published_namespace(namespace, published, transaction=None):
    """Bind published ``torch.*`` children to an independent root.

    Installers publish children while the compatibility owner is still the
    ``torch`` root. Once that root is replaced by :class:`TorchNamespace`,
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
    # ``core.install`` publishes ``torch.torch`` as an alias of the native
    # owner. Normalize only this explicit root alias; other published modules
    # may legitimately be shared implementation objects.
    owner_alias = "torch.torch"
    # Validate the complete parent closure before mutating the namespace. A
    # missing parent must never leave a partially published graph behind.
    missing = []
    for name in modules:
        parent_name = name.rsplit(".", 1)[0]
        if parent_name != "torch" and parent_name not in modules:
            missing.append((name, parent_name))
    if missing:
        name, parent_name = sorted(missing)[0]
        raise RuntimeError(
            "cannot bind published module %r: parent %r is not published"
            % (name, parent_name)
        )
    if modules.get(owner_alias) is namespace.owner:
        modules[owner_alias] = namespace
        old = published.get(owner_alias)
        published[owner_alias] = namespace
        if transaction is not None:
            transaction.record(published, owner_alias, old, namespace)
    parents = {"torch": namespace}
    for name in sorted(modules, key=lambda item: (item.count("."), item)):
        if name == "torch":
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = parents.get(parent_name)
        if parent is None:
            raise RuntimeError("published namespace parent disappeared: %r" % parent_name)
        if isinstance(parent, TorchNamespace):
            if transaction is not None:
                transaction.mutate_attr(parent, attr, modules[name])
            else:
                setattr(parent, attr, modules[name])
            parents[name] = modules[name]
            continue
        had_attr = attr in vars(parent)
        old = vars(parent).get(attr, _MISSING)
        if transaction is not None:
            def undo(parent=parent, attr=attr, had_attr=had_attr, old=old, expected=modules[name]):
                current = vars(parent).get(attr, _MISSING)
                if current is _MISSING and not had_attr:
                    return  # This owned insertion has already been removed.
                if current is not expected:
                    raise TransactionConflict("published parent binding changed externally: " + attr)
                if had_attr:
                    object.__setattr__(parent, attr, old)
                else:
                    object.__delattr__(parent, attr)
            transaction.record_undo(undo)
        object.__setattr__(parent, attr, modules[name])
        parents[name] = modules[name]
    return namespace


def publish_independent_namespace(namespace, registry, transaction=None):
    """Publish an independent namespace through one registry boundary.

    The runtime registry is the source of truth for the compatibility module
    graph.  Keeping the child binding and root replacement together prevents
    callers from publishing a detached ``torch`` root while leaving the
    registry's children attached to the native owner.  Every mutation is
    recorded in the caller's activation transaction when one is supplied.
    """

    if not isinstance(namespace, TorchNamespace):
        raise TypeError("namespace must be a TorchNamespace")
    published = getattr(registry, "_published", None)
    if published is None or not hasattr(published, "get"):
        raise TypeError("registry must expose a published module mapping")

    imports = getattr(registry, "_modules", None)
    self_alias = published.get("torch.torch")
    if imports is not None and (self_alias is namespace.owner or self_alias is namespace):
        imported_alias = imports.get("torch.torch", _MISSING)
        if imported_alias is not namespace.owner and imported_alias is not namespace:
            raise RuntimeError("torch.torch import alias differs from its registered owner")
    bind_published_namespace(namespace, published, transaction=transaction)
    # The root self-alias is part of the import graph as well as the registry.
    # Updating only the registry leaves torch.torch pointing at the native
    # owner and makes the next ownership validation reject our own install.
    if published.get("torch.torch") is namespace and imports is not None:
        old_alias = imports.get("torch.torch", _MISSING)
        if old_alias is not namespace.owner and old_alias is not namespace:
            raise RuntimeError("torch.torch import alias differs from its registered owner")
        if transaction is not None:
            transaction.record(imports, "torch.torch", old_alias, namespace)
        imports["torch.torch"] = namespace
    old = published.get("torch", _MISSING)
    if transaction is not None:
        transaction.record(published, "torch", old, namespace)
    published["torch"] = namespace
    return namespace


__all__ = [
    "TorchNamespace", "independent_torch_namespace", "namespace_owner",
    "bind_published_namespace", "publish_independent_namespace",
]
