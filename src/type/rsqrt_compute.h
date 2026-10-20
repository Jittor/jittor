#pragma once
#include "core/common.h"

#if defined(JIT_cuda) && !defined(IS_ACL)
namespace jittor {
// JIT kernels use --use_fast_math. Its rsqrtf flushes float32 subnormal inputs
// to signed zero. Dispatch those by bits, before any floating comparison, to
// double rsqrt so positive/negative subnormals keep finite/NaN semantics.
// Ordinary float32 inputs use CUDA rsqrtf, as the independent Torch kernel does.
inline __device__ float _rsqrtf(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0)
        return static_cast<float>(::rsqrt(static_cast<double>(value)));
    return ::rsqrtf(value);
}
} // namespace jittor
#endif
