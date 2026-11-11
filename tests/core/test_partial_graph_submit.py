import numpy as np
import pytest

import jittor as jt
from _helpers import capability as _test_capability


@pytest.fixture(params=["cpu", "cuda"])
def sync_device(request):
    device = request.param
    if device == "cuda" and not _test_capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    with jt.flag_scope(use_cuda=int(device == "cuda"), lazy_execution=1,
                       auto_flush_ops=0, async_flush_ops=0, keep_graph=0):
        yield device


def test_submit_pending_only_selected_roots_and_returns_identity():
    first = jt.array(np.array([1., 2.], dtype="float32")) + 1
    second = jt.array(np.array([3., 4.], dtype="float32")) * 2
    result = jt.submit_pending(first)
    assert result is first
    # The unrelated graph remains lazy until explicitly submitted.
    assert second.location() == "none"
    jt.submit_pending(second, device_sync=True)
    assert np.allclose(first.numpy(), [2., 3.])
    assert np.allclose(second.numpy(), [6., 8.])


def test_submit_pending_rejects_empty_and_non_var_roots():
    with pytest.raises(ValueError, match="at least one"):
        jt.submit_pending()
    with pytest.raises(TypeError, match="Var"):
        jt.submit_pending(object())


def test_ready_strong_sync_does_not_submit_empty_batches_or_siblings(sync_device):
    ready = jt.array(np.array([1., 2.], dtype="float32")) + 1
    ready.sync(False, False)
    sibling = jt.array(np.array([3., 4.], dtype="float32")) * 2
    before = jt.runtime.exec_called
    for _ in range(8):
        ready.sync(False, False)
    assert jt.runtime.exec_called == before
    assert sibling.location() == "none"
    assert ready.location() == ("device" if sync_device == "cuda" else "cpu")
    sibling.sync(False, False)
    assert jt.runtime.exec_called > before
    np.testing.assert_array_equal(ready.numpy(), [2., 3.])
    np.testing.assert_array_equal(sibling.numpy(), [6., 8.])


def test_ready_root_preserves_explicit_device_and_weak_syncs(sync_device):
    ready = jt.array(np.array([1., 2.], dtype="float32")) + 1
    ready.sync(False, False)
    for device_sync, weak_sync in [(True, False), (False, True)]:
        before = jt.runtime.exec_called
        ready.sync(device_sync, weak_sync)
        assert jt.runtime.exec_called > before
    np.testing.assert_array_equal(ready.numpy(), [2., 3.])


def test_ready_kept_root_still_executes_after_scope_exit(sync_device):
    with jt.flag_scope(keep_graph=1):
        ready = jt.array(np.array([1., 2.], dtype="float32")) + 1
        ready.sync(False, False)
    before = jt.runtime.exec_called
    ready.sync(False, False)
    assert jt.runtime.exec_called > before
    np.testing.assert_array_equal(ready.numpy(), [2., 3.])
