#pragma once
#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace at { namespace cuda { namespace philox { namespace detail {

inline __host__ __device__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState state) {
    return std::make_tuple(state.seed_, state.offset_);
}

}}}} // namespace at::cuda::philox::detail
