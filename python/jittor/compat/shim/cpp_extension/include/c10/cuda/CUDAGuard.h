// c10/cuda/CUDAGuard.h — RAII device guards. Host-only header (exts include it
// only under #ifndef __CUDACC__), so pulling <cuda_runtime.h> here is fine; the
// public torch/extension.h stays CUDA-free.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>   // jtorch::Device / optional<Device>

namespace c10 { namespace cuda {

// Sets the active CUDA device on construction, restores the previous one on
// destruction. Under CUDA_VISIBLE_DEVICES the visible device is index 0, which
// matches what jittor uses, so this is effectively a no-op but kept faithful.
struct CUDAGuard {
    int prev_ = -1;
    explicit CUDAGuard(int device) { set(device); }
    explicit CUDAGuard(jtorch::Device d) { set(d.is_cuda() ? d.index() : -1); }
    ~CUDAGuard() { if (prev_ >= 0) cudaSetDevice(prev_); }
    void set(int device) {
        if (device < 0) return;
        if (cudaGetDevice(&prev_) != cudaSuccess) { prev_ = -1; cudaGetLastError(); }
        cudaSetDevice(device);
    }
    CUDAGuard(const CUDAGuard&) = delete;
    CUDAGuard& operator=(const CUDAGuard&) = delete;
};

// Like CUDAGuard but the device is optional (no-op if nullopt / index < 0).
struct OptionalCUDAGuard {
    int prev_ = -1; bool armed_ = false;
    OptionalCUDAGuard() {}
    explicit OptionalCUDAGuard(int device) { set(device); }
    explicit OptionalCUDAGuard(jtorch::Device d) { set(d.is_cuda() ? d.index() : -1); }
    explicit OptionalCUDAGuard(jtorch::optional<jtorch::Device> d) {
        if (d.has_value() && d->is_cuda()) set(d->index());
    }
    ~OptionalCUDAGuard() { if (armed_ && prev_ >= 0) cudaSetDevice(prev_); }
    void set(int device) {
        if (device < 0) return;
        if (cudaGetDevice(&prev_) != cudaSuccess) { prev_ = -1; cudaGetLastError(); }
        cudaSetDevice(device); armed_ = true;
    }
    OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
    OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;
};

}} // namespace c10::cuda

namespace at { namespace cuda {
using c10::cuda::CUDAGuard;
using c10::cuda::OptionalCUDAGuard;
}}
