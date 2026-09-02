#pragma once
#include <tuple>
#include <ATen/cuda/PhiloxCudaState.h>

namespace at { namespace cuda { namespace philox {

// Torch returns the (seed, offset) pair a Philox counter is seeded from, choosing between
// immediate values and graph-capture pointers. Only the immediate form exists here.
inline __host__ __device__ std::tuple<uint64_t, uint64_t> unpack(const PhiloxCudaState& state) {
    return std::make_tuple(state.seed_, state.offset_);
}

namespace detail {
// The spelling this header used before it matched torch's namespace; kept so extensions written
// against it still compile.
using ::at::cuda::philox::unpack;
} // namespace detail

}}} // namespace at::cuda::philox
