"""Require native fallback accounting for every executed NPU test."""

import sys

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    if call.when in ("setup", "call") and call.excinfo is not None:
        error = call.excinfo.value
        item._npu_fallback_excinfo = (type(error), error, error.__traceback__)
    yield


@pytest.fixture(autouse=True)
def _forbid_backend_fallbacks(request):
    from jittor._runtime.fallback import forbid_backend_fallbacks

    scope = forbid_backend_fallbacks()
    scope.__enter__()
    try:
        yield
    finally:
        # Pytest resumes yield fixtures normally even when the test failed.
        error_info = sys.exc_info()
        if error_info[0] is None:
            error_info = getattr(
                request.node, "_npu_fallback_excinfo", (None, None, None))
        try:
            scope.__exit__(*error_info)
        finally:
            if hasattr(request.node, "_npu_fallback_excinfo"):
                del request.node._npu_fallback_excinfo
