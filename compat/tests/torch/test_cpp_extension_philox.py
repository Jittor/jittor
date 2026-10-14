"""CUDA extension Philox reservations share native checkpointed RNG state."""
import functools

import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>

__global__ void draw_kernel(float* output, uint64_t seed, uint64_t offset) {
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, 0, offset + threadIdx.x, &rng);
    output[threadIdx.x] = curand_uniform(&rng);
}
void draw(torch::Tensor output) {
    at::cuda::CUDAGuard guard{output.device()};
    auto generator = at::cuda::detail::getDefaultCUDAGenerator();
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto state = generator->philox_cuda_state(128);
    draw_kernel<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<float>(), state.seed_, state.offset_);
}
std::vector<uint64_t> reserve(uint64_t increment) {
    auto generator = at::cuda::detail::getDefaultCUDAGenerator();
    std::lock_guard<std::mutex> lock(generator->mutex_);
    auto state = generator->philox_cuda_state(increment);
    return {state.seed_, state.offset_};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("draw", &draw);
    m.def("reserve", &reserve);
}
'''


@functools.lru_cache(None)
def _extension():
    from torch.utils.cpp_extension import load_inline
    return load_inline("jittor_philox_checkpoint_probe", cuda_sources=_SOURCE,
                       extra_cuda_cflags=["-std=c++17"], verbose=False)


def _draw():
    output = torch.empty(128, device="cuda", dtype=torch.float32)
    _extension().draw(output)
    torch.cuda.synchronize()
    assert output.is_cuda
    return output.cpu().numpy().copy()


def test_extension_rng_checkpoint_and_reseed():
    with forbid_backend_fallbacks():
        torch.manual_seed(327)
        first = _draw()
        state = torch.cuda.get_rng_state()
        expected = _draw()
        assert not np.array_equal(first, expected)
        torch.cuda.set_rng_state(state)
        np.testing.assert_array_equal(_draw(), expected)
        torch.manual_seed(327)
        np.testing.assert_array_equal(_draw(), first)


def test_native_and_extension_rng_mixed_continuation():
    with forbid_backend_fallbacks():
        torch.manual_seed(739)
        torch.rand(37, device="cuda").cpu().numpy()
        _draw()
        state = torch.cuda.get_rng_state()
        def continuation():
            native = torch.randn(42, device="cuda").cpu().numpy().copy()
            extension = _draw()
            native_tail = torch.rand(39, device="cuda").cpu().numpy().copy()
            return native, extension, native_tail
        expected = continuation()
        torch.cuda.set_rng_state(state)
        for actual, wanted in zip(continuation(), expected):
            np.testing.assert_array_equal(actual, wanted)


def test_reservation_rounding_and_invalid_increment_are_atomic():
    torch.manual_seed(59)
    extension = _extension()
    assert extension.reserve(5) == [59, 0]
    assert extension.reserve(0) == [59, 8]
    before = torch.cuda.get_rng_state().numpy().copy()
    with pytest.raises((OverflowError, RuntimeError), match="too large"):
        extension.reserve(2**64 - 1)
    np.testing.assert_array_equal(torch.cuda.get_rng_state().numpy(), before)
    with pytest.raises((ValueError, RuntimeError), match="Invalid CUDA Philox"):
        jt.compile_extern.curand.curand_reserve_philox(-1)
    np.testing.assert_array_equal(torch.cuda.get_rng_state().numpy(), before)
