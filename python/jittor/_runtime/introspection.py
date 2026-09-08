"""Read-only observations of existing runtime services, never a second runtime.

Queries neither load optional libraries nor synchronize/collect pending graphs.
Snapshots are detached observations, not an atomic process-wide checkpoint.
"""

from collections import namedtuple
from collections.abc import Mapping
import operator
from typing import TYPE_CHECKING, FrozenSet, Sequence

from .capability import Capabilities, Capability, CapabilityState
from .flag_policy import RUNTIME_FLAGS, STARTUP_FLAGS
from .state import RuntimeContext, _snapshot_value

if TYPE_CHECKING:
    from typing_extensions import Protocol

    class _NativeObservations(Protocol):
        def known_backends(self) -> Sequence[str]: ...
        def registered_backends(self) -> Sequence[str]: ...
        def backend_device_count(self, name: str) -> int: ...
        def number_of_hold_vars(self) -> int: ...
        def number_of_lived_vars(self) -> int: ...
        def number_of_lived_ops(self) -> int: ...
        def async_launch_history(self, backend: str, device: int, stream: int) -> str: ...


class _ReadOnly:
    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError("introspection is read-only; use jt.runtime.scope for policy changes")

    def __delattr__(self, name):
        raise AttributeError("introspection is read-only")


class Device(namedtuple("Device", "backend index")):
    """A visible logical device; indices are backend-local, not physical IDs."""
    __slots__ = ()


class DeviceInventory(namedtuple("DeviceInventory", "capability devices")):
    """A backend capability and its visible devices, retaining failure evidence."""
    __slots__ = ()

    @property
    def count(self):
        """None if unprobed or failed; zero is never used to hide a failure."""
        if self.capability.failed or self.capability.unprobed:
            return None
        return len(self.devices)

    def __bool__(self):
        raise TypeError("inspect inventory.capability.enabled/failed and inventory.count explicitly")


class CapabilityQueries(_ReadOnly):
    """Backend/visible-device registry plus existing library capability evidence."""
    __slots__ = ("_capability", "_core")
    _capability: Capabilities
    _core: "_NativeObservations"

    def __init__(self, capability, core):
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_core", core)

    def backends(self):
        """Names known to this native core, including unregistered backends."""
        return tuple(self._core.known_backends())

    def registered_backends(self):
        return tuple(self._core.registered_backends())

    def devices(self, name):
        """Observe the named backend without switching current placement.

        Device counts come from that backend's registry callback, not the
        currently selected accelerator. Physical-presence evidence for an
        unregistered accelerator comes from the existing capability service.
        """
        if name not in self.backends():
            raise ValueError("unknown backend %r; known: %s" % (name, ", ".join(self.backends())))
        if name not in self.registered_backends():
            if name == "cpu":
                capability = Capability(name, "backend", CapabilityState.FAILED,
                                        "the mandatory CPU backend is not registered")
            else:
                original = self._capability.accelerator(name)
                state = CapabilityState.DISABLED if original.enabled else original.state
                capability = Capability(name, "backend", state,
                                        "backend is not registered: " + original.reason,
                                        dict(original.evidence, registered=False))
            return DeviceInventory(capability, ())
        try:
            count = operator.index(self._core.backend_device_count(name))
            if count < 0:
                raise ValueError("backend returned a negative device count: %d" % count)
        except Exception as error:
            # A driver/query error is data here, not an absent-device answer.
            capability = Capability(name, "backend", CapabilityState.FAILED,
                                    "device query failed: %s: %s" % (type(error).__name__, error),
                                    {"registered": True, "error_type": type(error).__name__})
            return DeviceInventory(capability, ())
        state = CapabilityState.AVAILABLE if count else CapabilityState.DISABLED
        capability = Capability(name, "backend", state,
                                "registered backend reports %d visible logical device(s)" % count,
                                {"registered": True, "visible_devices": count})
        return DeviceInventory(capability, tuple(Device(name, index) for index in range(count)))

    def backend(self, name):
        return self.devices(name).capability

    def libraries(self):
        return self._capability.libraries()

    def library(self, name):
        """Observe only; unlike jt.capability.library this cannot request a load."""
        return self._capability.library(name, load=False)


class PolicyValues(_ReadOnly, Mapping):
    """Live named policy values; mutable containers are returned frozen."""
    __slots__ = ("_source", "_names")
    _source: object
    _names: FrozenSet[str]

    def __init__(self, source, names):
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_names", frozenset(names))

    def __getitem__(self, name):
        if name not in self._names:
            raise KeyError(name)
        try:
            return _snapshot_value(getattr(self._source, name), immutable=True)
        except AttributeError:
            raise KeyError(name) from None

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self):
        return iter(sorted(name for name in self._names if hasattr(self._source, name)))

    def __len__(self):
        return sum(1 for _ in self)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self))

    def snapshot(self):
        return _snapshot_value(dict(self), immutable=True)


class PolicySnapshot(namedtuple("PolicySnapshot", "startup runtime")):
    __slots__ = ()


class EffectivePolicy(_ReadOnly):
    __slots__ = ("startup", "runtime")
    startup: PolicyValues
    runtime: PolicyValues

    def __init__(self, config, context):
        object.__setattr__(self, "startup", PolicyValues(config, STARTUP_FLAGS))
        object.__setattr__(self, "runtime", PolicyValues(context, RUNTIME_FLAGS))

    def snapshot(self):
        return PolicySnapshot(self.startup.snapshot(), self.runtime.snapshot())


class AllocatorCounters(namedtuple("AllocatorCounters", "enabled alloc_calls allocated_bytes free_calls freed_bytes")):
    """Cumulative stat-allocator counters, reset when its native mode changes.

    Bytes are allocation traffic, not current memory usage or process RSS.
    Disabled instrumentation is explicit; zero does not assert zero traffic.
    """
    __slots__ = ()


class CounterSnapshot(namedtuple("CounterSnapshot", "exec_calls allocator held_vars live_vars live_ops")):
    """An unsynchronized observation; creating it does not execute pending ops."""
    __slots__ = ()


class Counters(_ReadOnly):
    __slots__ = ("_context", "_core")
    _context: RuntimeContext
    _core: "_NativeObservations"

    def __init__(self, context, core):
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_core", core)

    @property
    def exec_calls(self):
        return int(self._context.exec_called)

    @property
    def allocator(self):
        context = self._context
        return AllocatorCounters(bool(context.use_stat_allocator),
                                 int(context.stat_allocator_total_alloc_call),
                                 int(context.stat_allocator_total_alloc_byte),
                                 int(context.stat_allocator_total_free_call),
                                 int(context.stat_allocator_total_free_byte))

    @property
    def held_vars(self):
        return int(self._core.number_of_hold_vars())

    @property
    def live_vars(self):
        return int(self._core.number_of_lived_vars())

    @property
    def live_ops(self):
        return int(self._core.number_of_lived_ops())

    def snapshot(self):
        return CounterSnapshot(self.exec_calls, self.allocator, self.held_vars,
                               self.live_vars, self.live_ops)


class Diagnostics(_ReadOnly):
    """Detached diagnostic text; querying never submits work or waits for a device."""
    __slots__ = ("_core",)
    _core: "_NativeObservations"

    def __init__(self, core):
        object.__setattr__(self, "_core", core)

    def launch_history(self, backend="cuda", device=0, stream=None):
        """Recent launch candidates, with None selecting all streams on a device.

        Candidates are not proof of which asynchronous operation caused a fault.
        Stream values are native stream handles, not ordinal stream indices.
        """
        if not isinstance(backend, str):
            raise TypeError("backend must be a name")
        device = operator.index(device)
        if device < 0:
            raise ValueError("device must be nonnegative")
        native_stream = -1 if stream is None else operator.index(stream)
        if stream is not None and native_stream < 0:
            raise ValueError("stream must be a nonnegative handle or None")
        return self._core.async_launch_history(backend, device, native_stream)


class Introspection(_ReadOnly):
    """The supported jt.introspection read-only capability/policy/counter API."""
    __slots__ = ("capabilities", "policy", "counters", "diagnostics")
    capabilities: CapabilityQueries
    policy: EffectivePolicy
    counters: Counters
    diagnostics: Diagnostics

    def __init__(self, capability, config, runtime, core):
        object.__setattr__(self, "capabilities", CapabilityQueries(capability, core))
        object.__setattr__(self, "policy", EffectivePolicy(config, runtime.context))
        object.__setattr__(self, "counters", Counters(runtime.context, core))
        object.__setattr__(self, "diagnostics", Diagnostics(core))


__all__ = ["Introspection", "CapabilityQueries", "Device", "DeviceInventory",
           "PolicyValues", "EffectivePolicy", "PolicySnapshot", "Counters",
           "AllocatorCounters", "CounterSnapshot", "Diagnostics"]
