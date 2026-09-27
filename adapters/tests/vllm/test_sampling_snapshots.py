"""Host contracts for immutable snapshots of vLLM's five sampling arrays.

Transfers are deliberately fake here: these tests cover publication, ownership,
and avoidable work. Real CUDA source-lifetime and model tests remain separate.
"""

import gc
import sys
import types
import uuid
import weakref

import numpy as np
import pytest

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import PATCHES


FIELDS = ('temperature', 'top_p', 'top_k', 'min_p', 'seeds')
PATH = 'vllm.v1.worker.gpu.sample.states'


@pytest.fixture
def harness(monkeypatch):
    runtime = types.SimpleNamespace(copies=0, synchronizations=0, device=0,
                                    pending=[], fail_next=False)

    class Tensor:
        pass

    def tensor(source, device, dtype):
        assert device == 'cuda'
        runtime.copies += 1
        if runtime.fail_next:
            runtime.fail_next = False
            raise RuntimeError('transfer failed')
        result = Tensor()
        result.device = runtime.device
        result.dtype = dtype
        # Read lazily, as a real asynchronous transfer may do.
        runtime.pending.append((result, source))
        return result

    def synchronize():
        runtime.synchronizations += 1
        for result, source in runtime.pending:
            result.values = np.array(source, dtype=result.dtype, copy=True)
        runtime.pending.clear()

    torch = types.ModuleType('torch')
    torch.Tensor = Tensor
    torch.tensor = tensor
    torch.cuda = types.SimpleNamespace(current_device=lambda: runtime.device,
                                       synchronize=synchronize)
    monkeypatch.setitem(sys.modules, 'torch', torch)

    class UvaBufferPool:
        def __init__(self, dtype):
            self.dtype = dtype

        def copy_to_uva(self, source):
            raise AssertionError('production explicit-transfer patch required')

    class Buffer:
        def __init__(self, values, dtype):
            self.np = np.array(values, dtype=dtype)
            self.pool = UvaBufferPool(dtype)
            self.gpu = None

        def copy_to_uva(self):
            self.gpu = self.pool.copy_to_uva(self.np)
            return self.gpu

    class SamplingStates:
        def __init__(self):
            for name in FIELDS:
                dtype = 'int64' if name == 'seeds' else ('int32' if name == 'top_k' else 'float32')
                setattr(self, name, Buffer([1, 2], dtype))

        def apply_staged_writes(self):
            # vLLM 0.24's method, including its submission order.
            self.temperature.copy_to_uva()
            self.top_p.copy_to_uva()
            self.top_k.copy_to_uva()
            self.min_p.copy_to_uva()
            self.seeds.copy_to_uva()

    module = types.SimpleNamespace(SamplingStates=SamplingStates)
    original = SamplingStates.apply_staged_writes
    hooks = []

    def install():
        with tx.runtime_hook('sampling-snapshot-test-' + uuid.uuid4().hex) as hook:
            PATCHES['vllm.v1.worker.gpu.buffer_utils'](
                types.SimpleNamespace(UvaBufferPool=UvaBufferPool))
            patch = PATCHES.get(PATH)
            if patch is not None:
                patch(module)
        hooks.append(hook)
        return hook

    install()
    yield types.SimpleNamespace(runtime=runtime, States=SamplingStates,
                                Buffer=Buffer, Pool=UvaBufferPool, install=install,
                                hooks=hooks, original=original, module=module)
    for hook in reversed(hooks):
        if hook.state != 'rolled_back':
            hook.rollback()


def test_unchanged_decode_metadata_needs_only_one_snapshot(harness):
    state = harness.States()
    for _ in range(32):
        state.apply_staged_writes()
    assert harness.runtime.copies == 5
    assert harness.runtime.synchronizations == 5


@pytest.mark.parametrize('field', FIELDS)
def test_same_numpy_object_updates_and_preserves_old_consumers(harness, field):
    state = harness.States()
    state.apply_staged_writes()
    buffer = getattr(state, field)
    source = buffer.np
    snapshots = [(buffer.gpu, source.copy())]
    for i in range(5):
        source[:] = [i + 10, i + 20]
        state.apply_staged_writes()
        snapshots.append((buffer.gpu, source.copy()))
        assert buffer.np is source
        # A caller can immediately reuse host storage after publication.
        source[:] = -1
    for gpu, expected in snapshots:
        np.testing.assert_array_equal(gpu.values, expected)
    assert harness.runtime.copies == 10


@pytest.mark.parametrize('change', ['buffer', 'pool', 'pool_dtype', 'source_dtype',
                                    'shape', 'strides', 'device', 'gpu'])
def test_replacement_and_metadata_changes_invalidate(harness, change):
    state = harness.States()
    state.apply_staged_writes()
    old = state.top_k.gpu
    if change == 'buffer':
        state.top_k = harness.Buffer([1, 2], 'int32')
    elif change == 'pool':
        state.top_k.pool = harness.Pool('int32')
    elif change == 'pool_dtype':
        state.top_k.pool.dtype = 'int64'
    elif change == 'source_dtype':
        state.top_k.np = state.top_k.np.astype('int64')
    elif change == 'shape':
        state.top_k.np = state.top_k.np.reshape(1, 2)
    elif change == 'strides':
        state.top_k.np = np.array([1, 99, 2, 99], dtype='int32')[::2]
    elif change == 'device':
        harness.runtime.device = 1
    else:
        state.top_k.gpu = object()
    state.apply_staged_writes()
    assert state.top_k.gpu is not old
    assert harness.runtime.copies == (10 if change == 'device' else 6)
    np.testing.assert_array_equal(state.top_k.gpu.values, state.top_k.np)
    np.testing.assert_array_equal(old.values, [1, 2])
    assert state.top_k.gpu.device == harness.runtime.device
    assert state.top_k.gpu.dtype == state.top_k.pool.dtype


def test_failed_update_is_not_cached_and_retries(harness):
    state = harness.States()
    state.apply_staged_writes()
    old = state.temperature.gpu
    state.temperature.np[:] = [7, 8]
    harness.runtime.fail_next = True
    with pytest.raises(RuntimeError, match='transfer failed'):
        state.apply_staged_writes()
    assert state.temperature.gpu is old
    state.apply_staged_writes()
    np.testing.assert_array_equal(state.temperature.gpu.values, [7, 8])
    np.testing.assert_array_equal(old.values, [1, 2])
    assert harness.runtime.copies == 7


def test_request_slot_reuse_restores_earlier_values(harness):
    state = harness.States()
    state.apply_staged_writes()
    original = state.seeds.gpu
    state.seeds.np[0] = 987654321
    state.apply_staged_writes()
    previous_request = state.seeds.gpu
    state.seeds.np[0] = 1
    state.apply_staged_writes()
    np.testing.assert_array_equal(state.seeds.gpu.values, [1, 2])
    np.testing.assert_array_equal(previous_request.values, [987654321, 2])
    np.testing.assert_array_equal(original.values, [1, 2])
    assert state.seeds.gpu is not original
    assert harness.runtime.copies == 7


def test_owner_isolation_and_lifetime(harness):
    first, second = harness.States(), harness.States()
    for state in (first, second, first, second):
        state.apply_staged_writes()
    assert first.top_k.gpu is not second.top_k.gpu
    assert harness.runtime.copies == 10
    owner_ref, gpu_ref = weakref.ref(first), weakref.ref(first.top_k.gpu)
    del first
    gc.collect()
    assert owner_ref() is None
    assert gpu_ref() is None


def test_direct_pool_and_other_buffers_still_publish_fresh_snapshots(harness):
    buffer = harness.Buffer([1, 2], 'int32')
    first = buffer.copy_to_uva()
    second = buffer.copy_to_uva()
    assert first is not second
    buffer.np[:] = -1
    np.testing.assert_array_equal(first.values, [1, 2])
    np.testing.assert_array_equal(second.values, [1, 2])
    assert harness.runtime.synchronizations == 2


def test_rollback_reinstall_drops_snapshot_cache(harness):
    state = harness.States()
    state.apply_staged_writes()
    previous = state.top_k.gpu
    harness.hooks[-1].rollback()
    assert harness.States.apply_staged_writes is harness.original
    assert not getattr(harness.States, '_jittor_sampling_snapshots', False)
    harness.install()
    state.apply_staged_writes()
    assert state.top_k.gpu is not previous
    state.apply_staged_writes()
    assert harness.runtime.copies == 10
    patch = PATCHES.get(PATH)
    assert patch is not None
    assert patch(harness.module) is False
