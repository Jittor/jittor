"""Assertion helpers used across the test suite."""

import re


def _exception_names(exc_type):
    types = exc_type if isinstance(exc_type, tuple) else (exc_type,)
    return " or ".join(item.__name__ for item in types)


def expect_error(func, *, exc_type=Exception, match=None):
    """Assert that ``func`` raises the requested exception and message.

    ``exc_type=Exception`` and ``match=None`` preserve legacy callers while
    they are migrated. New tests should always provide both arguments: merely
    observing that *something* failed lets an unrelated setup error satisfy an
    error-path test.
    """
    try:
        func()
    except exc_type as error:
        if match is not None and re.search(match, str(error)) is None:
            raise AssertionError(
                "{} message {!r} did not match {!r}".format(
                    type(error).__name__, str(error), match)
            ) from error
        return error
    except Exception as error:
        raise AssertionError(
            "expected {}, got {}: {}".format(
                _exception_names(exc_type), type(error).__name__, error)
        ) from error
    raise AssertionError(
        "did not raise {}".format(_exception_names(exc_type))
    )
