"""Private modules that implement the public package composition."""

from .registry import (
    BackendRegistry,
    BackendSpec,
    DuplicateRegistration,
    MissingKernel,
    OpRegistry,
    RegistryError,
    UnknownBackend,
)
from .fallback import BackendFallbackPolicy, FallbackDecision, FallbackError, FallbackMode

__all__ = [
    "BackendRegistry", "BackendSpec", "DuplicateRegistration",
    "MissingKernel", "OpRegistry", "RegistryError", "UnknownBackend",
    "BackendFallbackPolicy", "FallbackDecision", "FallbackError", "FallbackMode",
]
