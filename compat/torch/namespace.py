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


def native_module_facade(source, name):
    """Give native implementations an installation-owned writable namespace."""
    facade = types.ModuleType(name, source.__doc__)
    facade.__package__ = name if hasattr(source, "__path__") else name.rpartition(".")[0]
    if hasattr(source, "__path__"):
        facade.__path__ = []
    for key, value in vars(source).items():
        if key.startswith("__") and key != "__all__":
            continue
        if isinstance(value, (dict, list, set)):
            value = value.copy()
        setattr(facade, key, value)
    return facade


class TorchNamespace(types.ModuleType):
    """Module-shaped view over one Jittor compatibility owner.

    Missing reads may use the native owner while installers are migrated.
    Writes and deletions belong to this namespace. Native capabilities remain
    readable without making application patches mutate the native module.
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
        object.__setattr__(self, "_hidden_owner_names", frozenset())
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
        if name in self._LOCAL_METADATA or name in self._hidden_owner_names:
            raise AttributeError(name)
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        object.__setattr__(self, "_hidden_owner_names", self._hidden_owner_names - {name})

    def __delattr__(self, name):
        """Remove a local API without revealing a native fallback underneath."""
        if name.startswith("_") or name in self._LOCAL_METADATA:
            return super().__delattr__(name)
        if name in vars(self):
            super().__delattr__(name)
        elif name in self._hidden_owner_names or not hasattr(self.owner, name):
            raise AttributeError(name)
        object.__setattr__(self, "_hidden_owner_names", self._hidden_owner_names | {name})

    def __dir__(self):
        return sorted((set(super().__dir__()) | set(dir(self.owner))) - self._hidden_owner_names)

    def _binding_state(self, name):
        local = vars(self)
        return name in local, local.get(name), name in self._hidden_owner_names

    def _restore_binding(self, name, state):
        """Restore exact local ownership without mutating a fallback owner."""
        present, value, hidden = state
        if present:
            object.__setattr__(self, name, value)
        elif name in vars(self):
            object.__delattr__(self, name)
        if hidden:
            object.__setattr__(self, "_hidden_owner_names", self._hidden_owner_names | {name})
        else:
            object.__setattr__(self, "_hidden_owner_names", self._hidden_owner_names - {name})


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
