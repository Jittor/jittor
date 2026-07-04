#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

#define C10_CUDA_CHECK(EXPR) do { \
    cudaError_t _jt_cuda_err = (EXPR); \
    TORCH_CHECK(_jt_cuda_err == cudaSuccess, cudaGetErrorString(_jt_cuda_err)); \
} while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
