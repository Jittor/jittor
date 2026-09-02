#pragma once
#include <cstdint>

namespace at {

// The RNG state a CUDA kernel needs to generate dropout masks. Torch splits this out from
// CUDAGeneratorImpl.h so an extension can take the plain state without the generator; flash-attn
// includes it that way. Torch's own version also carries a graph-capture variant, which has no
// counterpart here -- Jittor does not capture CUDA graphs, so the state is always immediate
// values. It must stay trivially copyable: flash-attn placement-new's it into an opaque buffer
// inside its parameter struct and copies that struct to the device by value.
struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    __host__ __device__ PhiloxCudaState(uint64_t seed = 0, uint64_t offset = 0)
        : seed_(seed), offset_(offset) {}
};

} // namespace at
