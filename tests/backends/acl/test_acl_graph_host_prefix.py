"""A device recording must recompute explicit CPU prefixes on every call."""
from unittest.mock import patch

import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from _helpers.devices import explicit_tensor_placement as _placement
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor._runtime.graph_replay import GraphReplay


def _array(values, backend):
    with _placement(backend):
        value = jt.array(np.asarray(values, dtype="float32"))
    value.sync(True, False)
    return value


class _MaskNet(jt.nn.Module):
    def __init__(self, cpu_copy=False):
        super().__init__()
        self.cpu_copy = cpu_copy

    def execute(self, device_value, cpu_value):
        with _placement(0):
            if self.cpu_copy:
                cpu_value = jt.device_copy(cpu_value, -1)
            mask = jt.ternary(cpu_value > 0, cpu_value * 2, cpu_value - 3)
            other = cpu_value + 5
        return (device_value * mask.to_device(device_value.device_id)
                + other.to_device(device_value.device_id)) * 2


def _expected(device_value, cpu_value):
    mask = np.where(cpu_value > 0, cpu_value * 2, cpu_value - 3)
    return (device_value * mask + cpu_value + 5) * 2


@pytest.mark.npu
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("cpu_copy", [False, True])
def test_dynamic_cpu_prefix_and_two_boundaries_follow_every_input(queued, cpu_copy):
    require_accelerator("acl")
    before = jt.core.backend_fallback_count()
    source = np.arange(16, dtype="float32").reshape(4, 4) / 4
    x = _array(source, 2)
    feeds = [np.full((4, 4), n - 4, dtype="float32") for n in range(10)]
    host = [_array(values, 0) for values in feeds]
    replay = GraphReplay(_MaskNet(cpu_copy=cpu_copy))
    answers = []
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0, check_graph=1), forbid_backend_fallbacks():
            for value in host:
                result = replay(x, value)
                answers.append(result)
                if not queued:
                    result.sync(True, False)
            jt.sync(answers, True, False)
            assert replay.stats["graph"] > 0, replay._graph_refused
            assert replay._graph_has_host_prefix
            for actual, values in zip(answers, feeds):
                assert actual.location() == "device"
                np.testing.assert_array_equal(actual.numpy(), _expected(source, values))
            jt.core.graph_check()
    finally:
        replay.invalidate()
    assert replay._cuda_graph == 0
    assert not replay._graph_has_host_prefix
    assert jt.core.backend_fallback_count() == before


@pytest.mark.npu
def test_reused_cpu_source_mutation_is_copied_before_each_prefix():
    require_accelerator("acl")
    source = np.ones((4, 4), dtype="float32")
    x = _array(source, 2)
    host = _array(source, 0)
    replay = GraphReplay(_MaskNet())
    answers, expected = [], []
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0, check_graph=1), forbid_backend_fallbacks():
            for n in range(12):
                values = np.full((4, 4), n - 6, dtype="float32")
                host._write_inplace(values)
                answers.append(replay(x, host))
                expected.append(_expected(source, values))
            jt.sync(answers, True, False)
            assert replay.stats["graph"] > 0, replay._graph_refused
            for actual, want in zip(answers, expected):
                np.testing.assert_array_equal(actual.numpy(), want)
    finally:
        replay.invalidate()


@pytest.mark.npu
def test_changed_prefix_leaf_releases_recording_before_executor_fallback():
    require_accelerator("acl")
    source = np.ones((4, 4), dtype="float32")
    x, host = _array(source, 2), _array(source, 0)
    replay = GraphReplay(_MaskNet())
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0), forbid_backend_fallbacks():
            for _ in range(7):
                replay(x, host).sync(True, False)
            assert replay._graph_has_host_prefix, replay._graph_refused
            handle = replay._cuda_graph
            # A checked launch returning false must never be followed by a
            # launch of that stale handle. Ordinary executor replay answers.
            original_release = jt.core.graph_release
            released = []
            def release(value):
                released.append(value)
                return original_release(value)
            with patch.object(jt.core, "graph_launch_host_prefix", return_value=False), \
                    patch.object(jt.core, "graph_release", side_effect=release):
                result = replay(x, host)
                result.sync(True, False)
            assert released == [handle]
            assert replay._cuda_graph == 0
            np.testing.assert_array_equal(result.numpy(), _expected(source, source))
    finally:
        replay.invalidate()


@pytest.mark.npu
@pytest.mark.parametrize("kind", ["opaque", "random", "device_readback"])
def test_prefix_admission_rejects_opaque_random_and_device_dependencies(kind):
    require_accelerator("acl")
    source = np.ones((4, 4), dtype="float32")
    x = _array(source, 2)
    with jt.no_grad(), jt.flag_scope(auto_graph_replay=0, auto_flush_ops=0):
        with _placement(0):
            if kind == "opaque":
                value = jt.code((4, 4), "float32", [], cpu_src="for (int i=0;i<4;i++) for (int j=0;j<4;j++) @out(i,j)=1;")
                value = value._as_tensor()  # _is_constant is not proof of purity.
            elif kind == "random":
                value = jt.rand((4, 4))
            else:
                value = x.cpu() + 1
        output = x + value.to_device(x.device_id)
        assert jt.core.graph_host_prefix_bytes([output]) == -1
        output.sync(True, False)


@pytest.mark.npu
def test_prefix_retention_counts_cpu_and_device_storage():
    require_accelerator("acl")
    x = _array(np.ones((4, 4), dtype="float32"), 2)
    host = _array(np.ones((4, 4), dtype="float32"), 0)
    replay = GraphReplay(_MaskNet())
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0), forbid_backend_fallbacks():
            for _ in range(7):
                replay(x, host).sync(True, False)
            assert replay._graph_has_host_prefix, replay._graph_refused
            prefix_bytes = jt.core.graph_host_prefix_bytes(replay._capture.outputs)
            assert prefix_bytes >= 4 * 4 * 4 * 4
            replay._drop_recording()
            replay._graph_refused = None
            replay._max_retained_bytes = replay._graph_bytes + prefix_bytes - 1
            assert not replay._record_cuda_graph(replay._capture)
            assert "host prefix" in replay._graph_refused
            assert replay._cuda_graph == 0
    finally:
        replay.invalidate()


@pytest.mark.npu
def test_all_device_replay_uses_existing_launch_path():
    require_accelerator("acl")
    class DeviceOnly(jt.nn.Module):
        def execute(self, value):
            return value * 2 + 1
    source = np.arange(16, dtype="float32").reshape(4, 4)
    feed = [_array(source + n, 2) for n in range(10)]
    replay = GraphReplay(DeviceOnly())
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0), forbid_backend_fallbacks(), \
                patch.object(jt.core, "graph_launch_host_prefix", side_effect=AssertionError("unexpected host prefix")):
            for n, value in enumerate(feed):
                result = replay(value)
                result.sync(True, False)
                np.testing.assert_array_equal(result.numpy(), (source + n) * 2 + 1)
            assert replay.stats["graph"] > 0, replay._graph_refused
            assert not replay._graph_has_host_prefix
    finally:
        replay.invalidate()


def _host_copy_boundaries(outputs):
    seen, stack, found = set(), list(outputs), []
    while stack:
        value = stack.pop()
        if value.var_ptr in seen or value.is_finished:
            continue
        seen.add(value.var_ptr)
        inputs = value._inputs()
        if (value._producer_name() == "device_copy" and value.placement_backend == 2
                and inputs and inputs[0].placement_backend == 0):
            found.append(value)
        else:
            stack.extend(inputs)
    return found


@pytest.mark.npu
def test_real_boundary_migration_is_checked_after_prefix_recomputation():
    require_accelerator("acl")
    source = np.ones((64, 64), dtype="float32")
    x, host = _array(source, 2), _array(source, 0)
    replay = GraphReplay(_MaskNet())
    blockers = []
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0, check_graph=1), forbid_backend_fallbacks():
            for _ in range(7):
                replay(x, host).sync(True, False)
            assert replay._graph_has_host_prefix, replay._graph_refused
            boundary = _host_copy_boundaries(replay._capture.outputs)[0]
            old_address = boundary._mem_ptr_now
            handle = replay._cuda_graph
            # Use the supported migration API, never edit a pointer. Holding
            # the old device allocation with another tensor guarantees that
            # recomputation must allocate a different device destination.
            boundary.migrate_to_cpu()
            assert jt.core.graph_leaves_moved(handle)
            for _ in range(256):
                blocker = _array(np.zeros_like(source), 2)
                blockers.append(blocker)
                if blocker._mem_ptr_now == old_address:
                    break
            assert blockers[-1]._mem_ptr_now == old_address, "old block was not reused by allocation control"
            launches = jt.core.graph_launch_count()
            assert not jt.core.graph_launch_host_prefix(handle)
            assert boundary._mem_ptr_now != old_address
            assert jt.core.graph_leaves_moved(handle)
            assert jt.core.graph_launch_count() == launches
            # The ordinary wrapper now releases the stale graph and answers
            # through the executor; it cannot launch the old photograph.
            result = replay(x, host)
            result.sync(True, False)
            assert replay._cuda_graph == 0
            np.testing.assert_array_equal(result.numpy(), _expected(source, source))
    finally:
        replay.invalidate()
        blockers.clear()


def _held_bytes():
    return (jt.core.device_memory_used(-1)
            + sum(jt.core.device_memory_used(d) for d in range(jt.core.get_device_count())))


@pytest.mark.npu
def test_invalidate_releases_host_prefix_owners_and_storage():
    require_accelerator("acl")
    import gc
    source = np.ones((32, 32), dtype="float32")
    x, host = _array(source, 2), _array(source, 0)
    def exercise():
        replay = GraphReplay(_MaskNet())
        try:
            with jt.no_grad(), jt.flag_scope(auto_graph_replay=0), forbid_backend_fallbacks():
                for _ in range(9):
                    result = replay(x, host)
                    result.sync(True, False)
                assert replay._graph_has_host_prefix, replay._graph_refused
        finally:
            replay.invalidate()
        del result, replay
        gc.collect()
        jt.sync_all(True)
        jt.gc()
    exercise()  # Prime JIT/constant caches before taking the ownership baseline.
    before_bytes, before_holders = _held_bytes(), jt.number_of_hold_vars()
    exercise()
    assert jt.number_of_hold_vars() == before_holders
    assert _held_bytes() <= before_bytes + 256


@pytest.mark.npu
def test_low_level_prefix_launch_checks_boundaries_without_bind_leaves():
    require_accelerator("acl")
    source = np.ones((64, 64), dtype="float32")
    x, host = _array(source, 2), _array(source, 0)
    replay = GraphReplay(_MaskNet())
    handle, blockers = 0, []
    try:
        with jt.no_grad(), jt.flag_scope(auto_graph_replay=0, check_graph=1), forbid_backend_fallbacks():
            cap = replay._capture_now((x, host))
            replay._capture = cap
            with jt.flag_scope(keep_graph=2):
                assert jt.core.graph_capture_begin_host_prefix(cap.outputs)
                try:
                    jt.sync(cap.outputs, False, False)
                finally:
                    handle = jt.core.graph_capture_end()
            assert handle
            # Deliberately omit graph_bind_leaves: native prefix ownership has
            # to protect its own device boundary addresses independently.
            boundary = _host_copy_boundaries(cap.outputs)[0]
            old_address = boundary._mem_ptr_now
            boundary.migrate_to_cpu()
            for _ in range(256):
                blocker = _array(np.zeros_like(source), 2)
                blockers.append(blocker)
                if blocker._mem_ptr_now == old_address:
                    break
            assert blockers[-1]._mem_ptr_now == old_address
            assert not jt.core.graph_leaves_moved(handle)  # No optional binding exists.
            launches = jt.core.graph_launch_count()
            assert not jt.core.graph_launch_host_prefix(handle)
            assert boundary._mem_ptr_now != old_address
            assert jt.core.graph_launch_count() == launches
            # The old launch API also cannot bypass the checked prefix.
            with pytest.raises(RuntimeError, match="host prefix or graph leaf moved"):
                jt.core.graph_launch(handle)
            assert jt.core.graph_launch_count() == launches
    finally:
        if handle:
            jt.core.graph_release(handle)
        replay.invalidate()
        blockers.clear()
