"""Python flags, core state and dynamically compiled operators share storage."""

import ctypes
import ctypes.util
import functools
import threading

import jittor as jt
import numpy as np
import pytest


def _native_device_flags():
    return jt.code(
        [4], "int32", cpu_header='#include "runtime/runtime.h"',
        cpu_src='''
        auto& state = jittor::runtime_device_state();
        @out(0) = state.use_cuda;
        @out(1) = state.device_id;
        @out(2) = state.sync_run;
        @out(3) = &state == &jittor::native_runtime().devices()
            && &state.use_cuda == &jittor::runtime_flag_use_cuda()
            && &state.device_id == &jittor::runtime_flag_device_id()
            && &state.sync_run == &jittor::runtime_flag_sync_run();
        ''',
    ).numpy()


def test_python_flag_writes_reach_core_and_jit_owner():
    saved = jt.flags.sync_run
    with jt.flag_scope(use_cuda=0, sync_run=0):
        np.testing.assert_array_equal(
            _native_device_flags(), [0, jt.flags.device_id, 0, 1])
        with jt.flag_scope(sync_run=True):
            np.testing.assert_array_equal(
                _native_device_flags(), [0, jt.flags.device_id, 1, 1])
        assert jt.flags.sync_run == 0
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            assert all(getattr(jt.flags, name) == 0 for name in
                       ("use_cuda", "use_device", "use_acl", "use_rocm", "use_corex"))
    assert jt.flags.sync_run == saved


@functools.lru_cache(maxsize=None)
def _load_cudart():
    """libcudart, or None where it cannot be loaded.

    Asked for at run time, never during collection. A module-level
    ``_CUDART = _load_cudart()`` dlopen'd the CUDA runtime -- and ran
    ``ctypes.util.find_library``, which shells out to ``ldconfig``/``gcc`` -- on
    every bare import of this file, on every machine, whether or not the one
    test that needs the library ever runs. That is the collection-time backend
    side effect ``tests/structure/test_pytest_contract.py`` forbids, and the
    same rule the cross-device suites already follow
    (see ``tests/backends/cuda/test_device_copy.py``). ``lru_cache`` keeps the
    load to once per process, as the module-level constant did.
    """
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13",
                 ctypes.util.find_library("cudart")):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _thread_cuda_device(lib):
    """The calling thread's own CUDA device, straight from the runtime."""
    index = ctypes.c_int(-1)
    lib.cudaGetDevice(ctypes.byref(index))
    return index.value


def test_current_device_binds_the_calling_thread():
    """A thread that never called cudaSetDevice must still run where jittor says.

    CUDA's current device is per-host-thread and starts at 0 on every new one;
    jittor's lives in a single process-wide ``RuntimeDeviceState``. A pool
    worker that only reads the process value therefore ran on device 0 while
    everything that resolved a device through jittor -- an allocation, a memory
    query, a copy, an event -- was booked against device 1. That is a
    cross-device mismatch: its signature is a Xid 31 MMU fault and a
    context-sticky ``cudaErrorIllegalAddress`` that later surfaces on a call
    like ``cudaMemGetInfo``, which only reports that the context is gone. It hit
    rank 1 of a TP=2 run and never rank 0, whose threads default to the device
    it uses anyway.
    """
    cudart = _load_cudart()
    if cudart is None:
        pytest.skip("libcudart is not loadable here")
    if jt.get_device_count() < 2:
        pytest.skip("needs a second accelerator to distinguish 0 from the ambient one")
    target = 1
    saved = jt.flags.device_id
    seen = {}
    try:
        jt.flags.device_id = target
        assert jt.current_device() == target

        def worker():
            seen["reported"] = jt.current_device()
            seen["bound"] = _thread_cuda_device(cudart)

        thread = threading.Thread(target=worker, name="device-probe")
        thread.start()
        thread.join()
    finally:
        jt.flags.device_id = saved
    assert seen["reported"] == target
    assert seen["bound"] == target, (
        "a fresh thread was bound to device %d while jittor reported device %d, "
        "so every device operation it issued used the wrong context"
        % (seen["bound"], seen["reported"]))
