"""Retry decorator for tests that exercise nondeterministic tuning paths."""

from functools import wraps
import warnings


class RetryWarning(UserWarning):
    """Reports that a test needed retries or exhausted its retry budget."""


def retry(num):
    if num < 1:
        raise ValueError("retry count must allow at least one attempt")

    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            retries = 0
            inner.call_count += 1
            for attempt in range(1, num + 1):
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    if attempt == num:
                        inner.last_retries = retries
                        inner.total_retries += retries
                        if retries:
                            warnings.warn(
                                "%s failed after %d retries (%d attempts)"
                                % (func.__qualname__, retries, attempt),
                                RetryWarning,
                                stacklevel=2,
                            )
                        raise
                    retries += 1
                    continue

                inner.last_retries = retries
                inner.total_retries += retries
                if retries:
                    warnings.warn(
                        "%s passed after %d retries (%d attempts)"
                        % (func.__qualname__, retries, attempt),
                        RetryWarning,
                        stacklevel=2,
                    )
                return result

        inner.call_count = 0
        inner.last_retries = 0
        inner.total_retries = 0

        return inner

    return outer
