"""Fail-closed validation of the native runtime's backend fallback decisions."""

from contextlib import contextmanager


@contextmanager
def forbid_backend_fallbacks():
    """Reject cross-backend attempts, including exceptions swallowed by a caller.

    The caller must materialize or synchronize the work under validation inside
    this scope. No implicit device synchronization is performed here. An
    exception from the body propagates unchanged; the counter is checked only
    after a normal return.
    """
    import jittor as jt

    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(backend_fallback="error"):
        yield
        attempts = jt.core.backend_fallback_count() - before
        if attempts:
            raise RuntimeError(
                "backend fallback attempted %d time(s) inside forbidden scope"
                % attempts)


__all__ = ["forbid_backend_fallbacks"]
