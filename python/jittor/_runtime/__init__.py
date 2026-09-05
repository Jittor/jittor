"""Private modules that implement the public package composition."""

from .dispatch import (
    DispatchContext,
    KernelRegistration,
    dispatch_context,
    optional_kernel,
    override_kernel,
    register_kernel,
    registered_kernel,
    select_kernel,
    try_dispatch,
    unregister_kernel,
)
from .fallback import BackendFallbackPolicy, FallbackDecision, FallbackError, FallbackMode

__all__ = [
    "DispatchContext", "KernelRegistration", "dispatch_context", "optional_kernel",
    "override_kernel", "register_kernel", "registered_kernel", "select_kernel",
    "try_dispatch", "unregister_kernel",
    "BackendFallbackPolicy", "FallbackDecision", "FallbackError", "FallbackMode",
]
