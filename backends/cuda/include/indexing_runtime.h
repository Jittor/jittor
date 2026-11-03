#pragma once

// Runtime calls for generated CUDA-family indexing kernels. Keep SDK
// selection in a header: KernelIR processes the indexing source body before
// the device compiler sees it.
#ifdef IS_ROCM
#include <hip/hip_runtime.h>
#include "runtime/backend.h"
#include "utils/log.h"
#else
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {
inline void indexing_copy_device(void* output, const void* input, size_t bytes, int device) {
#ifdef IS_ROCM
    auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(device));
    auto status = hipMemcpyAsync(output, input, bytes, hipMemcpyDeviceToDevice, stream);
    CHECK(status == hipSuccess) << "HIP indexed update copy: " << hipGetErrorString(status);
#else
    checkCudaErrors(cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
#endif
}
} // namespace jittor
