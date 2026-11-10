"""Fresh CUDA writes preserve storage and prior readers without a global drain."""
import numpy as np
import ctypes
from pathlib import Path
import pytest
import jittor as jt
import torch
from _helpers.common import device_flags_for, get_all_device_types


@pytest.fixture(params=get_all_device_types())
def device(request):
    with jt.flag_scope(**device_flags_for(request.param), auto_flush_ops=0):
        yield request.param


def _allocated(device, dtype=torch.float32):
    data = np.arange(12, dtype=np.float32).reshape(3, 4) % 4 - 1
    x = torch.tensor(data, device=device, dtype=dtype)
    value = x @ x.T
    value.sync()
    return value, data @ data.T


def _count_global_sync(monkeypatch):
    original = jt.sync_all
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(jt, 'sync_all', counted)
    return calls


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_allocated_update_preserves_readers_without_global_drain(device, dtype, monkeypatch):
    with torch.no_grad():
        value, expected = _allocated(device, dtype)
        # Observe internal allocation identity without exporting a pointer to
        # an external consumer; the public data_ptr path is tested below.
        pointer = int(value._mem_ptr_now)
        token, version = value._saved_storage_version()[1:]
        snapshot = value.clone()
        prior_reader = value * 2
        rhs = value + 0.5
        calls = _count_global_sync(monkeypatch)
        assert value.copy_(rhs) is value
        assert len(calls) == (0 if device == 'cuda' else 1)
        assert value._saved_storage_version()[1:] == [token, version + 1]
        np.testing.assert_array_equal(value.float().numpy(), expected + 0.5)
        np.testing.assert_array_equal(snapshot.float().numpy(), expected)
        np.testing.assert_array_equal(prior_reader.float().numpy(), expected * 2)
        assert value.data_ptr() == pointer
        assert value.data_ptr() != rhs.data_ptr()
        assert value.location() == ('device' if device == 'cuda' else 'cpu')
        rhs.add_(3)
        np.testing.assert_array_equal(value.float().numpy(), expected + 0.5)


@pytest.mark.parametrize('kind', ['detach', 'data', 'view', 'transpose', 'holder'])
def test_existing_alias_keeps_imperative_write(device, kind, monkeypatch):
    with torch.no_grad():
        value, expected = _allocated(device)
        if kind == 'detach':
            alias = value.detach()
        elif kind == 'data':
            alias = value.data
        elif kind == 'view':
            alias = value.view(9)
        elif kind == 'transpose':
            alias = value.T
        else:
            alias = value._alias()
        calls = _count_global_sync(monkeypatch)
        value.add_(1)
        assert len(calls) == 1
        np.testing.assert_array_equal(value.numpy(), expected + 1)
        alias_expected = (expected + 1).T if kind == 'transpose' else expected + 1
        np.testing.assert_array_equal(alias.numpy(), alias_expected.reshape(tuple(alias.shape)))


def test_grad_enabled_call_keeps_imperative_write(device, monkeypatch):
    with torch.enable_grad():
        value, expected = _allocated(device)
        calls = _count_global_sync(monkeypatch)
        value.add_(2)
        assert len(calls) == 1
        np.testing.assert_array_equal(value.numpy(), expected + 2)


def test_fresh_writes_survive_async_auto_flush(device):
    with torch.no_grad():
        value, expected = _allocated(device)
        pointer = int(value._mem_ptr_now)
        snapshots = []
        with jt.flag_scope(auto_flush_ops=4):
            for index in range(12):
                snapshots.append(value.clone())
                value.add_(0.5)
        for index, snapshot in enumerate(snapshots):
            np.testing.assert_array_equal(snapshot.numpy(), expected + index * 0.5)
        np.testing.assert_array_equal(value.numpy(), expected + 6)
        assert value.data_ptr() == pointer


@pytest.mark.parametrize('through_alias', [False, True])
@pytest.mark.parametrize('accessor', ['data_ptr', 'device_raw_ptr', 'device_ptr_ready'])
def test_exported_cuda_pointer_keeps_external_reads_current(device, accessor, through_alias, monkeypatch):
    if device != 'cuda':
        pytest.skip('external device pointer test requires CUDA')
    # Resolve the runtime selected by this build, and copy through its raw API
    # without calling Jittor sync: an external consumer cannot see a lazy write.
    libraries = sorted(Path(jt.compiler.cuda_lib).glob('libcudart.so*'))
    assert libraries, 'the active CUDA build must provide libcudart'
    runtime = ctypes.CDLL(str(libraries[0]))
    runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    runtime.cudaMemcpy.restype = ctypes.c_int
    with torch.no_grad():
        value, expected = _allocated(device)
        exporter = value.detach() if through_alias else value
        pointer = exporter.data_ptr() if accessor == 'data_ptr' else int(getattr(exporter, accessor))
        # A raw pointer can outlive the Python alias that exported it.
        del exporter
        calls = _count_global_sync(monkeypatch)
        value.add_(1)
        assert len(calls) == 1
        host = np.empty_like(expected)
        assert runtime.cudaMemcpy(host.ctypes.data, pointer, host.nbytes, 2) == 0
        np.testing.assert_array_equal(host, expected + 1)


def test_foreign_dlpack_storage_keeps_external_owner_current(device, monkeypatch):
    from jittor.compat.torch.frontend import tensor_frontend

    with torch.no_grad():
        owner, expected = _allocated(device)
        with tensor_frontend(torch.Tensor, device=device):
            borrowed = jt.from_dlpack(jt.to_dlpack(owner))
        assert isinstance(borrowed, torch.Tensor)
        calls = _count_global_sync(monkeypatch)
        borrowed.add_(1)
        assert len(calls) == 1
        np.testing.assert_array_equal(owner.numpy(), expected + 1)
        np.testing.assert_array_equal(borrowed.numpy(), expected + 1)
