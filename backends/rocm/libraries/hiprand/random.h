#pragma once
#include <hip/hip_runtime_api.h>
#include <cstdint>

namespace jittor {
hipError_t hiprand_generate(void* output, int64_t count, bool use_double, bool normal,
                           uint64_t seed, uint64_t offset, hipStream_t stream);
} // namespace jittor
