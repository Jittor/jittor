// c10/cuda/CUDAGuard.h — RAII device guards. Host-only header (exts include it
// only under #ifndef __CUDACC__), so pulling <cuda_runtime.h> here is fine; the
// public torch/extension.h stays CUDA-free.
#pragma once
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>   // jtorch::Device / optional<Device>

namespace c10 { namespace cuda {

// Sets the active CUDA device on construction, restores the previous one on
// destruction.
//
// It also has to move jittor's *own* current device. The shim's tensor factories
// -- `torch::empty` and friends, which an extension uses for its out /
// rng_state / softmax_lse_accum / out_accum buffers -- build their Var from
// jittor's current device, and `cudaSetDevice` does not move that one. Left out
// of step, a cuda:1 extension allocates those buffers on cuda:0 and hands their
// pointers to cuda:1 kernels: cudaErrorIllegalAddress.
struct CUDAGuard {
    int prev_ = -1;
    int prev_accel_ = -1;
    explicit CUDAGuard(int device) { set(device); }
    explicit CUDAGuard(jtorch::Device d) { set(d.is_cuda() ? d.index() : -1); }
    ~CUDAGuard() {
        if (prev_ >= 0) cudaSetDevice(prev_);
        if (prev_accel_ >= 0) jtorch::detail::accelerator_set_current_device(prev_accel_);
    }
    void set(int device) {
        if (device < 0) return;
        if (cudaGetDevice(&prev_) != cudaSuccess) { prev_ = -1; cudaGetLastError(); }
        cudaSetDevice(device);
        prev_accel_ = jtorch::detail::accelerator_current_device();
        jtorch::detail::accelerator_set_current_device(device);
    }
    CUDAGuard(const CUDAGuard&) = delete;
    CUDAGuard& operator=(const CUDAGuard&) = delete;
};

// Like CUDAGuard but the device is optional (no-op if nullopt / index < 0).
struct OptionalCUDAGuard {
    int prev_ = -1; bool armed_ = false;
    int prev_accel_ = -1;
    OptionalCUDAGuard() {}
    explicit OptionalCUDAGuard(int device) { set(device); }
    explicit OptionalCUDAGuard(jtorch::Device d) { set(d.is_cuda() ? d.index() : -1); }
    explicit OptionalCUDAGuard(jtorch::optional<jtorch::Device> d) {
        if (d.has_value() && d->is_cuda()) set(d->index());
    }
    ~OptionalCUDAGuard() {
        if (armed_ && prev_ >= 0) cudaSetDevice(prev_);
        if (armed_ && prev_accel_ >= 0) jtorch::detail::accelerator_set_current_device(prev_accel_);
    }
    void set(int device) {
        if (device < 0) return;
        if (cudaGetDevice(&prev_) != cudaSuccess) { prev_ = -1; cudaGetLastError(); }
        cudaSetDevice(device); armed_ = true;
        prev_accel_ = jtorch::detail::accelerator_current_device();
        jtorch::detail::accelerator_set_current_device(device);
    }
    OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
    OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;
};

}} // namespace c10::cuda

namespace at { namespace cuda {
using c10::cuda::CUDAGuard;
using c10::cuda::OptionalCUDAGuard;
}}
