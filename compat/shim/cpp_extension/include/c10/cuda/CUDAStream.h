// c10/cuda/CUDAStream.h — the stream a torch extension built on this shim should
// launch on.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>

namespace c10 { namespace cuda {

struct CUDAStream {
    cudaStream_t s_;
    CUDAStream(cudaStream_t s = (cudaStream_t)0) : s_(s) {}
    cudaStream_t stream() const { return s_; }
    operator cudaStream_t() const { return s_; }
};

// jittor's compute stream.
//
// This header used to say "jittor launches on the CUDA default stream (0)" and
// return 0. That stopped being true when jittor moved to `cudaStreamPerThread`
// (`compute_stream` in `backends/cuda/runtime/driver.cc`), and the difference is
// not cosmetic: the runtime's per-thread stream and the legacy default stream do
// NOT synchronise with each other, so an extension kernel launched on 0 is an
// unordered straggler. jittor schedules its own ops -- and its allocator's frees
// -- with no ordering against it, and can hand an operand or an output buffer to
// something else while the kernel is still reading it. That is a cross-device
// access waiting to happen: observed as a `cudaErrorIllegalAddress` out of an
// extension kernel on the rank whose device is not the process default.
//
// `cudaStreamPerThread` is the constant 0x2 (alias `CU_STREAM_PER_THREAD` in the
// driver API); neither API has a call that returns it. Set
// JITTOR_SHIM_LEGACY_STREAM=1 to launch on the legacy default stream again.
inline cudaStream_t shim_launch_stream() {
#ifdef cudaStreamPerThread
    return getenv("JITTOR_SHIM_LEGACY_STREAM") ? (cudaStream_t)0 : cudaStreamPerThread;
#else
    return getenv("JITTOR_SHIM_LEGACY_STREAM") ? (cudaStream_t)0 : (cudaStream_t)0x2;
#endif
}

inline CUDAStream getCurrentCUDAStream(int = -1) { return CUDAStream(shim_launch_stream()); }
inline CUDAStream getDefaultCUDAStream(int = -1) { return CUDAStream(shim_launch_stream()); }
// A *side* stream request still gets the legacy stream on purpose. The right
// target for it is one of jittor's own side streams (`jt._cuda_stream_handle`),
// not the compute stream -- returning the compute stream here would silently
// serialise whatever the caller meant to overlap -- and no launch path in this
// shim's users asks for one.
inline CUDAStream getStreamFromPool(bool = false, int = -1) { return CUDAStream((cudaStream_t)0); }

}} // namespace c10::cuda

namespace at { namespace cuda {
using c10::cuda::CUDAStream;
using c10::cuda::getCurrentCUDAStream;
using c10::cuda::getDefaultCUDAStream;
using c10::cuda::getStreamFromPool;
}} // namespace at::cuda
