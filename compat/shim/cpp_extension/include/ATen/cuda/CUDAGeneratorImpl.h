#pragma once
#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>
#include <limits>
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>

namespace jittor {
std::vector<long long> curand_reserve_philox(long long increment);
}

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
    PhiloxCudaState philox_cuda_state(uint64_t increment) {
        if (increment > static_cast<uint64_t>(std::numeric_limits<long long>::max()))
            throw std::overflow_error("CUDA Philox increment is too large");
        auto span = jittor::curand_reserve_philox(static_cast<long long>(increment));
        return PhiloxCudaState(static_cast<uint64_t>(span.at(0)),
                               static_cast<uint64_t>(span.at(1)));
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
