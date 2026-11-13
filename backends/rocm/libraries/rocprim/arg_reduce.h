#pragma once
#include <hip/hip_runtime_api.h>
#include "scan.h"

namespace jittor {
hipError_t rocm_arg_reduce(const void* input, const int* offsets, int rows,
                          void* values, int* indices, RocprimScanType dtype,
                          bool minimum, hipStream_t stream);
} // namespace jittor
