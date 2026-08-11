"""Assertion helpers used across the test suite."""


def expect_error(func):
    try:
        func()
    except Exception:
        return
    raise Exception("Expect an error, but nothing catched.")
