"""Contract for the first read-only Runtime/Context state view."""

import pytest


def test_runtime_sync_run_is_a_live_read_only_view_of_the_native_flag():
    import jittor as jt

    original = jt.flags.sync_run
    try:
        assert jt.runtime.sync_run == original
        with jt.flag_scope(sync_run=0):
            assert jt.runtime.sync_run == 0
        assert jt.runtime.sync_run == original

        with pytest.raises(AttributeError):
            jt.runtime.sync_run = 0
    finally:
        jt.flags.sync_run = original


def test_runtime_state_does_not_duplicate_device_or_backend_flags():
    import jittor as jt

    assert tuple(jt.runtime.__slots__) == ("_context",)
    assert jt.runtime.context.__class__.__name__ == "RuntimeContext"
    assert jt.runtime.context._flags is jt.flags
    assert jt.runtime.context.snapshot() == {
        "sync_run": jt.flags.sync_run,
        "device_id": getattr(jt.flags, "device_id", -1),
        "use_cuda": jt.flags.use_cuda,
        "lazy_execution": jt.flags.lazy_execution,
    }
    assert jt.runtime.device_id == getattr(jt.flags, "device_id", -1)
    assert jt.runtime.use_cuda == jt.flags.use_cuda


def test_runtime_context_is_the_single_sync_run_owner():
    import jittor as jt

    context = jt.runtime.context
    original = jt.flags.sync_run
    try:
        jt.flags.sync_run = 0
        assert context.sync_run == 0
        assert jt.runtime.sync_run == 0
        with pytest.raises(AttributeError):
            context.sync_run = 1
        with pytest.raises(AttributeError):
            jt.runtime.context = context
    finally:
        jt.flags.sync_run = original


def test_runtime_device_id_is_a_live_read_only_view():
    import jittor as jt

    assert jt.runtime.device_id == getattr(jt.flags, "device_id", -1)
    with pytest.raises(AttributeError):
        jt.runtime.device_id = 0


def test_runtime_use_cuda_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.use_cuda
    try:
        assert jt.runtime.use_cuda == original
        with jt.flag_scope(use_cuda=0):
            assert jt.runtime.use_cuda == 0
        assert jt.runtime.use_cuda == original
        with pytest.raises(AttributeError):
            jt.runtime.use_cuda = 0
    finally:
        jt.flags.use_cuda = original


def test_runtime_lazy_execution_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.lazy_execution
    try:
        assert jt.runtime.lazy_execution == original
        with jt.flag_scope(lazy_execution=0):
            assert jt.runtime.lazy_execution == 0
            assert jt.runtime.context.snapshot()["lazy_execution"] == 0
        assert jt.runtime.lazy_execution == original
        with pytest.raises(AttributeError):
            jt.runtime.lazy_execution = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.lazy_execution = 0
    finally:
        jt.flags.lazy_execution = original
