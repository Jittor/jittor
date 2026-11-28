#pragma once

#ifdef IS_ROCM
#include <hip/hip_bfloat16.h>

namespace jittor {

// HIP's storage-compatible BF16 class has an explicit float constructor,
// whereas generated Jittor kernels rely on ordinary floating-point
// assignment at materialization boundaries. Keep HIP rounding and the
// native 16-bit layout, and expose the converting constructor.
struct bfloat16 : hip_bfloat16 {
    __host__ __device__ bfloat16() = default;
    __host__ __device__ bfloat16(float value) : hip_bfloat16(value) {}
    __host__ __device__ bfloat16(hip_bfloat16 value) : hip_bfloat16(value) {}
};

static_assert(sizeof(bfloat16) == sizeof(hip_bfloat16), "ROCm BF16 storage size");
static_assert(alignof(bfloat16) == alignof(hip_bfloat16), "ROCm BF16 storage alignment");

} // namespace jittor
#endif
