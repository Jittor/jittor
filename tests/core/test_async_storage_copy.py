"""Explicit storage transfers: layout, lifetime, direction and stream ordering."""
import gc

import jittor as jt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def default_cpu():
    with jt.flag_scope(use_cuda=0):
        yield


def ready(values, device=False):
    with jt.flag_scope(use_cuda=int(device)):
        value = jt.array(np.asarray(values, dtype=np.float32)).stop_grad()
        value.sync()
    return value


def empty(shape, device=False):
    with jt.flag_scope(use_cuda=int(device)):
        value = jt.empty(shape, dtype="float32").stop_grad()
        value.sync(False, False)
    return value


@pytest.fixture
def accelerator():
    if not (jt.has_cuda or getattr(jt, "has_acl", False)):
        pytest.skip("accelerator build required")
    with jt.flag_scope(use_cuda=1, backend_fallback="error", auto_graph_replay=0):
        probe = (jt.array([2.]) + 1).stop_grad()
        probe.sync()
        assert probe.location() == "device"
        before = jt.core.backend_fallback_count()
        yield
        assert jt.core.backend_fallback_count() == before


def test_cpu_copy_and_wait_preserve_destination_address():
    src, dst = ready([2, 4, 6]), empty((3,))
    address = dst._storage_address
    jt.core._copy_into_async(dst, src)
    jt.core._copy_wait(dst)
    jt.core._copy_wait(dst)
    assert dst._storage_address == address
    np.testing.assert_array_equal(dst.numpy(), [2, 4, 6])


@pytest.mark.parametrize("source_view", [True, False])
@pytest.mark.parametrize("expanded", [False, True])
def test_reject_strided_storage_without_touching_bytes(source_view, expanded):
    base = ready(np.arange(8))
    view = base[:2].reshape((2, 1)).expand(2, 2) if expanded else base[1::2]
    view.sync()
    assert not view._storage_is_contiguous()
    dense = ready([10, 20, 30, 40])
    dst, src = (dense, view) if source_view else (view, dense)
    with pytest.raises(RuntimeError, match="contiguous"):
        jt.core._copy_into_async(dst, src)
    np.testing.assert_array_equal(base.numpy(), np.arange(8))
    np.testing.assert_array_equal(dense.numpy(), [10, 20, 30, 40])


def test_reject_overlap_and_allow_disjoint_offset_views():
    base = ready(np.arange(10))
    a, b = base[:4], base[1:5]
    a.sync(); b.sync()
    with pytest.raises(RuntimeError, match="non-overlapping"):
        jt.core._copy_into_async(b, a)
    dst = base[6:10]; dst.sync()
    jt.core._copy_into_async(dst, a)
    jt.core._copy_wait(base)
    np.testing.assert_array_equal(base.numpy(), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    jt.core._copy_into_async(base, base)


def test_reject_size_and_unmaterialized_input():
    dst, src = empty((4,)), ready([1, 2])
    with pytest.raises(RuntimeError, match="equal byte sizes"):
        jt.core._copy_into_async(dst, src)
    pending = jt.ones(4) + 7
    with pytest.raises(RuntimeError, match="materialized"):
        jt.core._copy_into_async(dst, pending)


def test_empty_copy():
    src, dst = empty((0,)), empty((0,))
    jt.core._copy_into_async(dst, src)
    jt.core._copy_wait(dst)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
def test_device_copy_directions(accelerator, direction):
    values = np.arange(65539, dtype=np.float32) * 0.25
    src = ready(values, direction != "h2d")
    dst = empty(values.shape, direction != "d2h")
    address = dst._storage_address
    jt.core._copy_into_async(dst, src)
    jt.core._copy_wait(dst)
    assert dst._storage_address == address
    assert dst.location() == ("cpu" if direction == "d2h" else "device")
    np.testing.assert_array_equal(dst.numpy(), values)


def test_temporary_source_and_destination_rebind(accelerator):
    values = np.arange(4 * 1024 * 1024, dtype=np.float32)
    src = ready(values)
    dst = empty(values.shape, True)
    alias = dst.detach().reshape((1024, -1)); alias.sync(False, False)
    address = dst._storage_address
    jt.core._copy_into_async(dst, src)
    del src
    dst.assign(empty(values.shape, True))
    gc.collect()
    pressure = [ready(np.full(values.shape, -7, dtype=np.float32)) for _ in range(2)]
    jt.core._copy_wait(alias)
    assert alias._storage_address == address
    np.testing.assert_array_equal(alias.numpy().reshape(-1), values)


def test_device_producer_order_and_repeated_copies(accelerator):
    values = np.arange(1024 * 1024, dtype=np.float32)
    src = ready(values, True)
    dst = empty(values.shape, True)
    for i in range(3):
        with jt.flag_scope(use_cuda=1):
            computed = (src * 2 + i).stop_grad()
            computed.sync(False, False)
        jt.core._copy_into_async(dst, computed)
        jt.core._copy_wait(dst)
        np.testing.assert_array_equal(dst.numpy(), values * 2 + i)
        # numpy() may migrate storage: use a new physical device destination.
        dst = empty(values.shape, True)


def test_byte_write_updates_shared_mutation_version():
    src, dst = ready([3, 4]), ready([1, 2])
    alias = dst.detach()
    alias.sync(False, False)
    before = alias._saved_storage_version()
    jt.core._copy_into_async(dst, src)
    jt.core._copy_wait(alias)
    after = alias._saved_storage_version()
    assert after[1] == before[1]
    assert after[2] == before[2] + 1
    np.testing.assert_array_equal(alias.numpy(), [3, 4])


def test_reject_state_capture_without_writing():
    src, dst = ready([3, 4]), ready([1, 2])
    jt.core._state_capture_begin()
    try:
        with pytest.raises(RuntimeError, match="capture"):
            jt.core._copy_into_async(dst, src)
        with pytest.raises(RuntimeError, match="capture"):
            jt.core._copy_wait(dst)
    finally:
        jt.core._state_capture_end()
    np.testing.assert_array_equal(dst.numpy(), [1, 2])
