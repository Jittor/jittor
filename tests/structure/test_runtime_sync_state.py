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

    assert tuple(jt.runtime.__slots__) == ()
    assert not hasattr(jt.runtime, "use_cuda")
    assert not hasattr(jt.runtime, "device_id")
