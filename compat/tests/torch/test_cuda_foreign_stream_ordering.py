"""Foreign CUDA work must share Jittor's real compute-stream ordering."""

import ctypes
import threading
import time

import pytest

from _helpers import capability as _test_capability


@pytest.mark.parametrize("stream_kind", ["current", "default", "new"])
def test_public_stream_event_waits_for_native_compute(stream_kind):
    import jittor as jt
    import torch

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("real CUDA stream ordering requires a CUDA device")
    driver = ctypes.CDLL("libcuda.so.1")

    def checked(name, *args):
        result = getattr(driver, name)(*args)
        assert result == 0, (name, result)

    with jt.flag_scope(use_cuda=1):
        value = torch.zeros((8,), dtype=torch.int32, device="cuda")
        pointer = ctypes.c_uint64(value.data_ptr())
        torch.cuda.synchronize()
        stream = {"current": torch.cuda.current_stream,
                  "default": torch.cuda.default_stream,
                  "new": torch.cuda.Stream}[stream_kind]()
        native_stream = ctypes.c_void_p(2)  # backend CUDA compute_stream/PTDS
        observed_stream = ctypes.c_void_p(stream.cuda_stream)
        event = ctypes.c_void_p()
        checked("cuEventCreate", ctypes.byref(event), 2)
        entered, release = threading.Event(), threading.Event()
        expired = []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

        @callback_type
        def gate(_):
            entered.set()
            if not release.wait(10):
                expired.append(True)

        try:
            checked("cuLaunchHostFunc", native_stream, gate, None)
            assert entered.wait(5), "native stream gate did not start"
            # Queue an actual device write behind the native compute gate.
            checked("cuMemsetD32Async", pointer, ctypes.c_uint(7),
                    ctypes.c_size_t(8), native_stream)
            checked("cuEventRecord", event, observed_stream)
            deadline = time.monotonic() + .1
            status = 600  # CUDA_ERROR_NOT_READY
            while time.monotonic() < deadline:
                status = driver.cuEventQuery(event)
                if status == 0:
                    break
                assert status == 600, status
                time.sleep(.002)
            assert status == 600, (
                "public stream event finished while native compute was still gated",
                stream.cuda_stream)
        finally:
            release.set()
            checked("cuCtxSynchronize")
            checked("cuEventDestroy_v2", event)
        assert not expired, "native stream gate timed out"
        assert value.tolist() == [7] * 8
