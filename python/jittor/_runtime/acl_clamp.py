"""Small runtime hook for the optional ACL clamp implementation.

The ACL compiler is loaded after the core Python API.  Keeping this hook in a
runtime-owned module avoids publishing a backend callback as a private
attribute on the public :mod:`jittor` module.  The hook deliberately returns
``None`` when ACL is unavailable; the caller then uses the portable clamp
implementation.
"""

from __future__ import absolute_import


_handler = None


def register_acl_clamp(handler):
    """Install *handler* and return the previously registered handler.

    ``handler`` must be callable.  Replacement is intentional: backend
    discovery may run more than once during a process lifetime.
    """
    if not callable(handler):
        raise TypeError("ACL clamp handler must be callable")
    global _handler
    previous = _handler
    _handler = handler
    return previous


def unregister_acl_clamp(handler=None):
    """Remove the ACL clamp handler and return the handler that was removed.

    If *handler* is supplied, only that exact registration is removed.  This
    makes cleanup safe when a later backend has already replaced an older one.
    """
    global _handler
    previous = _handler
    if handler is None or handler is _handler:
        _handler = None
    return previous if previous is not None and (handler is None or handler is previous) else None


def dispatch_acl_clamp(*args, **kwargs):
    """Dispatch to ACL when registered, otherwise return ``None``."""
    handler = _handler
    if handler is None:
        return None
    return handler(*args, **kwargs)


# Short names keep backend call sites readable while the explicit names above
# remain the stable contract used by tests and future runtime consumers.
register = register_acl_clamp
unregister = unregister_acl_clamp
dispatch = dispatch_acl_clamp


__all__ = [
    "dispatch_acl_clamp",
    "register_acl_clamp",
    "unregister_acl_clamp",
    "dispatch",
    "register",
    "unregister",
]
