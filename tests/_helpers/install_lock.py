"""Probe the compatibility install lock from a thread that does not own it.

Asking ``InstallTransaction._lock.acquire(blocking=False)`` on the calling
thread proves nothing about a leak. The lock is an ``RLock``, so the very thread
that leaked an acquire re-acquires it happily and reports success -- which is
how a leaked acquire in the install failure path stayed green under a test that
looked like it was checking for exactly that. A leak is only observable from
another thread.
"""

import threading

from jittor.compat.transaction import InstallTransaction


def install_lock_is_free(timeout=2.0):
    """Whether a thread other than this one can take and release the lock."""
    outcome = []

    def probe():
        acquired = InstallTransaction._lock.acquire(timeout=timeout)
        outcome.append(acquired)
        if acquired:
            # Release from the acquiring thread: an RLock refuses a release from
            # anywhere else, and doing it in the caller would raise instead of
            # reporting.
            InstallTransaction._lock.release()

    thread = threading.Thread(target=probe, name="install-lock-probe")
    thread.start()
    thread.join(timeout + 5.0)
    return bool(outcome) and outcome[0]
