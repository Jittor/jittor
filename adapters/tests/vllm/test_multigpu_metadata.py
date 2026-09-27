"""Request metadata must follow the worker's current CUDA device."""

import ctypes

import numpy as np
import pytest


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("uva", [False, True])
def test_staged_request_writes_follow_worker_device(index, uva):
    import torch
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    if torch.cuda.device_count() < 2:
        pytest.skip("insufficient-devices: requires two CUDA devices")
    previous = torch.cuda.current_device()
    torch.cuda.set_device(index)
    try:
        state = StagedWriteTensor((2, 8), dtype=torch.int32,
                                  device=torch.device('cuda', index),
                                  uva_instead_of_gpu=uva)
        shim = hasattr(torch, '_torch_compat_install_context')
        # Native UVA is mapped host memory accessible to both GPUs; this
        # vLLM build labels that view cuda:0 even when rank 1 consumes it.
        # Jittor instead allocates device memory, which must follow the rank.
        if not uva or shim:
            assert state.gpu.device == torch.device('cuda', index)
        state.stage_write(0, 1, [11, 12])
        state.stage_write(1, 3, [17])
        state.apply_write()
        torch.cuda.synchronize()
        expected = np.zeros((2, 8), dtype=np.int32)
        expected[0, 1:3] = [11, 12]
        expected[1, 3] = 17
        np.testing.assert_array_equal(state.gpu.cpu().numpy(), expected)
        state.stage_write(0, 2, [23])
        state.apply_write()
        torch.cuda.synchronize()
        expected[0, 2] = 23
        np.testing.assert_array_equal(state.gpu.cpu().numpy(), expected)
        assert torch.cuda.current_device() == index
        if shim:
            # Jittor stages a real device copy; native vLLM's UVA may instead
            # expose mapped host memory, so only the copy contract uses this.
            driver = ctypes.CDLL('libcuda.so.1')
            pointer = ctypes.c_uint64(state.gpu.data_ptr())
            memory, ordinal = ctypes.c_uint(), ctypes.c_int()
            assert driver.cuPointerGetAttribute(ctypes.byref(memory), 2, pointer) == 0
            assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, pointer) == 0
            assert memory.value == 2
            assert ordinal.value == index
    finally:
        torch.cuda.set_device(previous)
