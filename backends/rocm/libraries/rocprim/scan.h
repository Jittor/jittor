#pragma once

#include <cstddef>
#include <hip/hip_runtime_api.h>

namespace jittor {

enum class RocprimScanType { Float32, Float64, Int32, Int64 };

hipError_t rocprim_scan(void* workspace, size_t& workspace_bytes,
                       const void* input, void* output, size_t count,
                       RocprimScanType dtype, bool reverse, hipStream_t stream);

} // namespace jittor
