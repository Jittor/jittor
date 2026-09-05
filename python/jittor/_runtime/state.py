"""Startup configuration and runtime policy, independent of native bootstrap."""

from collections.abc import Mapping
from types import MappingProxyType, ModuleType

from .flag_policy import FLAG_ALIASES, READONLY_FLAGS, RUNTIME_FLAGS, STARTUP_FLAGS


def _snapshot_value(value, immutable=False):
    if isinstance(value, Mapping):
        values = {key: _snapshot_value(item, immutable) for key, item in value.items()}
        return MappingProxyType(values) if immutable else values
    if isinstance(value, (list, tuple)):
        values = [_snapshot_value(item, immutable) for item in value]
        return tuple(values) if immutable else values
    return value


class StartupConfig:
    """Detached, immutable configuration captured after backend bootstrap."""

    __slots__ = ("_values",)

    def __init__(self, native_flags):
        values = {name: _snapshot_value(getattr(native_flags, name), True)
                  for name in sorted(STARTUP_FLAGS) if hasattr(native_flags, name)}
        object.__setattr__(self, "_values", MappingProxyType(values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("jt.config is immutable; set startup options before import")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._values))

    def snapshot(self):
        return {name: _snapshot_value(value) for name, value in self._values.items()}


class _FrozenCompilerModule(ModuleType):
    def __setattr__(self, name, value):
        if name in STARTUP_FLAGS:
            raise AttributeError(name + " is immutable startup configuration")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in STARTUP_FLAGS:
            raise AttributeError(name + " is immutable startup configuration")
        super().__delattr__(name)


def freeze_compiler_config(module):
    # Keep module identity and its initialized values; reject late public
    # assignments that would otherwise diverge Python builds from native JIT.
    module.__class__ = _FrozenCompilerModule


class RuntimeContext:
    """Read-only diagnostic access to the current native policy and counters."""

    __slots__ = ("_flags",)

    def __init__(self, native_flags):
        object.__setattr__(self, "_flags", native_flags)

    def __getattr__(self, name):
        canonical = FLAG_ALIASES.get(name, name)
        if canonical not in RUNTIME_FLAGS | READONLY_FLAGS:
            raise AttributeError(name)
        if canonical == "device_id":
            return getattr(self._flags, canonical, -1)
        return getattr(self._flags, canonical)

    def __setattr__(self, name, value):
        raise AttributeError("runtime.context is read-only; use jt.runtime")

    def __dir__(self):
        return sorted(set(super().__dir__()) | {
            name for name in RUNTIME_FLAGS | READONLY_FLAGS | FLAG_ALIASES.keys()
            if hasattr(self._flags, FLAG_ALIASES.get(name, name))
        })

    def snapshot(self):
        return {name: _snapshot_value(getattr(self, name))
                for name in sorted(RUNTIME_FLAGS | READONLY_FLAGS) if hasattr(self, name)}


class RuntimeState:
    """Live runtime switches backed by native setters, plus read-only counters."""

    __slots__ = ("_context", "_scope_factory")

    def __init__(self, context, scope_factory=None):
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_scope_factory", scope_factory)

    def __getattr__(self, name):
        return getattr(self._context, name)

    def _writable_name(self, name):
        canonical = FLAG_ALIASES.get(name, name)
        if canonical in STARTUP_FLAGS:
            raise AttributeError(name + " is startup configuration; use jt.config")
        if canonical in READONLY_FLAGS:
            raise AttributeError(name + " is a read-only runtime counter")
        if canonical not in RUNTIME_FLAGS:
            raise AttributeError(name)
        getattr(self._context._flags, canonical)
        return canonical

    def __setattr__(self, name, value):
        setattr(self._context._flags, self._writable_name(name), value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._context)))

    @property
    def context(self):
        return self._context

    def snapshot(self):
        return self._context.snapshot()

    def scope(self, **changes):
        for name in changes:
            self._writable_name(name)
        if self._scope_factory is None:
            raise RuntimeError("runtime scope is unavailable before composition")
        return self._scope_factory(**changes)


__all__ = ["StartupConfig", "RuntimeContext", "RuntimeState"]
