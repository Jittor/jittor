#pragma once
#include <cuda_runtime.h>
#include "runtime/cuda_streams.h"

namespace jittor {
inline cudaStream_t cuda_side_stream(CudaSideStreamKind kind, int device) {
    return reinterpret_cast<cudaStream_t>(backend_stream(
        {accelerator_backend_id(), device}, backend_side_kind(kind)).handle);
}
inline cudaStream_t cuda_compute_stream(int device) {
    return reinterpret_cast<cudaStream_t>(backend_stream(
        {accelerator_backend_id(), device}, BackendStreamKind::Compute).handle);
}
} // namespace jittor
