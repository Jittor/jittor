// c10/cuda/CUDAStream.h — jittor launches on the CUDA default stream (0).
#pragma once
#include <cuda_runtime.h>

namespace c10 { namespace cuda {

struct CUDAStream {
    cudaStream_t s_;
    CUDAStream(cudaStream_t s = (cudaStream_t)0) : s_(s) {}
    cudaStream_t stream() const { return s_; }
    operator cudaStream_t() const { return s_; }
};

inline CUDAStream getCurrentCUDAStream(int = -1) { return CUDAStream((cudaStream_t)0); }
inline CUDAStream getStreamFromPool(bool = false, int = -1) { return CUDAStream((cudaStream_t)0); }
inline CUDAStream getDefaultCUDAStream(int = -1) { return CUDAStream((cudaStream_t)0); }

}} // namespace c10::cuda

namespace at { namespace cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getDefaultCUDAStream;
using c10::cuda::getStreamFromPool;
}} // namespace at::cuda
