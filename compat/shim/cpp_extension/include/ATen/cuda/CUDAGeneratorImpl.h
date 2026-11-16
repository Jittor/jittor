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
    bool captured_ = false;
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    int64_t* seed_ptr_ = nullptr;
    int64_t* offset_ptr_ = nullptr;
    uint32_t intragraph_offset_ = 0;
    __host__ __device__ PhiloxCudaState(uint64_t seed = 0, uint64_t offset = 0)
        : seed_(seed), offset_(offset) {}
    __host__ __device__ PhiloxCudaState(int64_t* seed, int64_t* offset, uint32_t increment)
        : captured_(true), seed_ptr_(seed), offset_ptr_(offset), intragraph_offset_(increment) {}
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
    if (state.captured_)
        return std::make_tuple(static_cast<uint64_t>(*state.seed_ptr_),
            static_cast<uint64_t>(*state.offset_ptr_) + state.intragraph_offset_);
    return std::make_tuple(state.seed_, state.offset_);
}
}} // namespace cuda::philox

} // namespace at
