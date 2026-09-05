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


__all__ = ["TorchNamespace", "independent_torch_namespace", "namespace_owner"]
