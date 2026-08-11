#pragma once
#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>

namespace at {

struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    __host__ __device__ PhiloxCudaState(uint64_t seed = 0, uint64_t offset = 0)
        : seed_(seed), offset_(offset) {}
};

struct Generator {};

struct CUDAGeneratorImpl : public Generator {
    std::mutex mutex_;
    PhiloxCudaState philox_cuda_state(uint64_t offset) {
        return PhiloxCudaState(0, offset);
    }
};

template <typename T>
T* get_generator_or_default(std::optional<Generator>, T* default_gen) {
    return default_gen;
}

namespace cuda { namespace detail {
inline CUDAGeneratorImpl* getDefaultCUDAGenerator() {
    static CUDAGeneratorImpl gen;
    return &gen;
}
}} // namespace cuda::detail

namespace cuda { namespace philox {
inline __host__ __device__ std::tuple<uint64_t, uint64_t> unpack(const PhiloxCudaState& state) {
    return std::make_tuple(state.seed_, state.offset_);
}
}} // namespace cuda::philox

} // namespace at
