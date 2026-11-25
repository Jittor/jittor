"""Shared fail-closed scope for real ROCm arithmetic tests."""

import pytest


@pytest.fixture(autouse=True)
def native_rocm():
    import jittor as jt
    from jittor._runtime import forbid_backend_fallbacks

    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0):
        with forbid_backend_fallbacks():
            yield
            jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before
