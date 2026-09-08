"""Independent Torch namespace with explicit native implementation delegation.

The frontend owns its module identity, types and public writes. Installation
onto the native Jittor module is rejected by the activation boundary.
"""

from __future__ import annotations

import importlib.machinery
import types
from typing import Any, cast


def native_module_facade(source, name):
    """Copy materialized public values without triggering native lazy imports."""
    facade = types.ModuleType(name, source.__doc__)
    facade.__package__ = name if hasattr(source, "__path__") else name.rpartition(".")[0]
    if hasattr(source, "__path__"):
        facade.__path__ = []
    source_values = vars(source)
    names = source_values.get("__all__")
    if names is None:
        names = tuple(key for key in source_values if not key.startswith("_"))
    exported = []
    for key in names:
        if not isinstance(key, str) or key.startswith("_"):
            continue
        # Looking up a legacy lazy export can import a private implementation
        # module and attach it to the native parent (linalg.complex is one).
        # Copy existing bindings only. native_api declares and installs the
        # supported Torch mathematical delegates separately.
        if key not in source_values:
            continue
        value = source_values[key]
        if isinstance(value, types.ModuleType):
            continue
        if isinstance(value, (dict, list, set)):
            value = value.copy()
        setattr(facade, key, value)
        exported.append(key)
    cast(Any, facade).__all__ = tuple(exported)
    return facade


class TorchNamespace(types.ModuleType):
    """Module-shaped view over one Jittor compatibility owner.

    Bootstrap may read the backend while installers collect their delegates.
    Once sealed, reads, writes and deletions stay within explicit bindings.
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
        object.__setattr__(self, "_sealed", False)
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
        if name in self._LOCAL_METADATA or name in self._hidden_owner_names or self._sealed:
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
        elif self._sealed or name in self._hidden_owner_names or not hasattr(self.owner, name):
            raise AttributeError(name)
        object.__setattr__(self, "_hidden_owner_names", self._hidden_owner_names | {name})

    def __dir__(self):
        if self._sealed:
            return sorted(set(super().__dir__()) - self._hidden_owner_names)
        return sorted((set(super().__dir__()) | set(dir(self.owner))) - self._hidden_owner_names)

    def _seal(self):
        """End implicit backend reads after all explicit API bindings exist."""
        object.__setattr__(self, "_sealed", True)

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
