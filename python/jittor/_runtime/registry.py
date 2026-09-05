"""Small, dependency-free backend and operator registries.

This module is the first migration seam for the multi-backend design.  It is
deliberately independent from the native runtime: importing it does not load
``jittor`` or probe a device.  Existing ``flags.use_cuda`` and C++ operator
registration therefore keep their current ownership until a later migration
stage can hand them over atomically.
"""

from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


def _cpu_device_count() -> int:
    """Return the logical host-device count used by the CPU provider."""
    return 1


def _cpu_allocator(size: int) -> bytearray:
    """Allocate zeroed host storage for the registry's CPU provider.

    This deliberately returns a Python-owned buffer: the registry remains
    importable without loading the native runtime while still exposing a real
    allocator hook that callers can exercise.
    """
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("CPU allocation size must be an integer")
    if size < 0:
        raise ValueError("CPU allocation size must be non-negative")
    return bytearray(size)


class RegistryError(RuntimeError):
    """Base class for registry failures."""


class DuplicateRegistration(RegistryError, ValueError):
    """Raised when a name is registered twice without replacement."""


class UnknownBackend(RegistryError, KeyError):
    """Raised when an operation references an unknown backend."""


class MissingKernel(RegistryError, LookupError):
    """Raised when a backend has no implementation for an operation."""


@dataclass(frozen=True)
class BackendSpec:
    """Capabilities and hooks owned by one backend.

    Hooks are optional at this stage.  ``device_count`` is a callable so a
    future native backend can report devices without rebuilding the registry.
    """

    name: str
    device_count: Callable[[], int] = lambda: 0
    allocator: Optional[Callable[[int], Any]] = None
    set_device: Optional[Callable[[int], None]] = None
    memcpy: Optional[Callable[..., Any]] = None
    synchronize: Optional[Callable[[], None]] = None
    stream: Optional[Callable[[], Any]] = None
    capabilities: Mapping[str, bool] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, False))


class BackendRegistry:
    """Thread-safe registry keyed by backend name.

    ``default()`` is lazy and returns CPU/CUDA entries with stable capability
    metadata.  CUDA's device count remains zero until a native provider fills
    it in; registering the entry still makes availability queryable rather
    than conflating "known backend" with "hardware present".
    """

    _default: Optional["BackendRegistry"] = None
    _default_lock = threading.Lock()

    def __init__(self, specs: Iterable[BackendSpec] = ()):
        self._lock = threading.RLock()
        self._specs: Dict[str, BackendSpec] = {}
        for spec in specs:
            self.register(spec)

    @classmethod
    def default(cls) -> "BackendRegistry":
        if cls._default is None:
            with cls._default_lock:
                if cls._default is None:
                    cls._default = cls((
                        BackendSpec(
                            "cpu", device_count=_cpu_device_count,
                            allocator=_cpu_allocator,
                            capabilities={"allocator": True, "memcpy": True,
                                          "synchronize": True}),
                        BackendSpec(
                            "cuda", capabilities={"allocator": True,
                                                   "memcpy": True,
                                                   "synchronize": True,
                                                   "stream": True}),
                    ))
        return cls._default

    def register(self, spec: BackendSpec, *, replace: bool = False) -> BackendSpec:
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("backend name must be a non-empty string")
        with self._lock:
            if spec.name in self._specs and not replace:
                raise DuplicateRegistration("backend already registered: %s" % spec.name)
            self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> BackendSpec:
        with self._lock:
            try:
                return self._specs[name]
            except KeyError as exc:
                raise UnknownBackend(name) from exc

    def names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._specs)

    def backend_for(self, value: Any) -> str:
        """Resolve the backend owning a runtime value.

        Native ``Var`` objects expose ``location()``; accepting that narrow
        protocol keeps this registry independent from the native module while
        giving dispatchers one canonical device-to-backend decision.
        """
        location = getattr(value, "location", None)
        if callable(location):
            location = location()
        # Native CPU Vars report ``location() == "none"`` (the device is
        # implicit), while lightweight callers commonly use ``cpu``/``host``.
        if location in (None, "none", "cpu", "host"):
            return "cpu"
        # Providers may expose either a concrete device (``cuda:0``) or the
        # backend-level location (``cuda``) while selecting a device later.
        # Both names identify the same dispatch backend; malformed/unknown
        # locations must continue to fail closed below.
        if location == "cuda" or (
            isinstance(location, str) and location.startswith("cuda:")
        ):
            return "cuda"
        raise UnknownBackend("cannot resolve backend for location: %r" % (location,))

    def supported_ops(self, op_registry: "OpRegistry", backend: str) -> Tuple[str, ...]:
        self.get(backend)
        return op_registry.supported_ops(backend)


class OpRegistry:
    """Dispatch table keyed by ``(operator id, backend name)``."""

    def __init__(self, backends: Optional[BackendRegistry] = None):
        self.backends = backends or BackendRegistry.default()
        self._lock = threading.RLock()
        self._kernels: Dict[Tuple[str, str], Callable[..., Any]] = {}

    _default: Optional["OpRegistry"] = None
    _default_lock = threading.Lock()

    @classmethod
    def default(cls) -> "OpRegistry":
        if cls._default is None:
            with cls._default_lock:
                if cls._default is None:
                    cls._default = cls()
        return cls._default

    def register(self, op: str, backend: str, kernel: Callable[..., Any], *,
                 replace: bool = False) -> Callable[..., Any]:
        if not op or not isinstance(op, str):
            raise ValueError("operator id must be a non-empty string")
        self.backends.get(backend)
        if not callable(kernel):
            raise TypeError("kernel must be callable")
        key = (op, backend)
        with self._lock:
            if key in self._kernels and not replace:
                raise DuplicateRegistration("kernel already registered: %s/%s" % key)
            self._kernels[key] = kernel
        return kernel

    def resolve(self, op: str, backend: str) -> Callable[..., Any]:
        self.backends.get(backend)
        with self._lock:
            try:
                return self._kernels[(op, backend)]
            except KeyError as exc:
                raise MissingKernel("no kernel registered for %s/%s" % (op, backend)) from exc

    def has_kernel(self, op: str, backend: str) -> bool:
        """Return whether a backend currently owns an implementation.

        Providers use this probe during teardown instead of catching a lookup
        exception.  Backend validation is retained so a misspelled backend is
        still reported as an ``UnknownBackend`` rather than looking absent.
        """
        self.backends.get(backend)
        with self._lock:
            return (op, backend) in self._kernels

    def unregister(self, op: str, backend: str) -> Callable[..., Any]:
        """Remove and return one kernel during backend/provider teardown."""
        self.backends.get(backend)
        key = (op, backend)
        with self._lock:
            try:
                return self._kernels.pop(key)
            except KeyError as exc:
                raise MissingKernel("no kernel registered for %s/%s" % key) from exc

    def dispatch(self, op: str, backend: str, *args: Any, **kwargs: Any) -> Any:
        return self.resolve(op, backend)(*args, **kwargs)

    def dispatch_value(self, op: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch using the backend selected from the first runtime value."""
        backend = self.backends.backend_for(value)
        return self.dispatch(op, backend, value, *args, **kwargs)

    def supported_ops(self, backend: str) -> Tuple[str, ...]:
        self.backends.get(backend)
        with self._lock:
            return tuple(sorted(op for op, bk in self._kernels if bk == backend))
