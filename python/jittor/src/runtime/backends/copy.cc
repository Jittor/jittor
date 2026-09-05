#include "runtime/backends/copy.h"
#include <cstring>
#include <exception>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "runtime/device.h"
#include "runtime/cuda_streams.h"
#endif

namespace jittor {

void cpu_backend_copy(void* dst, Device dst_device, const void* src,
                      Device src_device, size_t size, bool) {
    CHECK(dst_device.backend == BackendId::Cpu && src_device.backend == BackendId::Cpu)
        << "CPU copy requires host memory";
    if (size) std::memcpy(dst, src, size);
}

void cpu_backend_copy_async(void* dst, Device dst_device, const void* src,
                            Device src_device, size_t size, BackendStream) {
    cpu_backend_copy(dst, dst_device, src, src_device, size, false);
}

#ifdef HAS_CUDA
namespace {

auto copy_kind(Device dst, Device src) {
    if (src.backend == BackendId::Cpu) return cudaMemcpyHostToDevice;
    if (dst.backend == BackendId::Cpu) return cudaMemcpyDeviceToHost;
    #ifdef IS_CUDA
    // Unified addressing handles peer and driver-staged cross-device copies.
    if (src.index != dst.index) return cudaMemcpyDefault;
    #endif
    return cudaMemcpyDeviceToDevice;
}

template<class Func>
void on_copy_device(int device, Func&& func) {
    int previous = current_device();
    if (previous != device) set_current_device(device);
    try {
        func();
    } catch (...) {
        auto failure = std::current_exception();
        try {
            if (previous >= 0 && current_device() != previous) set_current_device(previous);
        } catch (...) {
            LOGe << "Could not restore device after backend copy failed";
        }
        std::rethrow_exception(failure);
    }
    if (previous >= 0 && current_device() != previous) set_current_device(previous);
}

} // namespace

void accelerator_backend_copy_async(void* dst, Device dst_device, const void* src,
                                    Device src_device, size_t size, BackendStream stream) {
    if (!size) return;
    on_copy_device(stream.device.index, [&] {
        checkCudaErrors(cudaMemcpyAsync(dst, src, size, copy_kind(dst_device, src_device),
                                       reinterpret_cast<cudaStream_t>(stream.handle)));
    });
}

void accelerator_backend_copy(void* dst, Device dst_device, const void* src,
                              Device src_device, size_t size, bool ordered) {
    if (!size) return;
    int device = dst_device.backend == BackendId::Cpu ? src_device.index : dst_device.index;
    on_copy_device(device, [&] {
        if (src_device.backend != BackendId::Cpu && dst_device.backend != BackendId::Cpu
                && (ordered || src_device.index != dst_device.index)) {
            int source = src_device.index;
            if (source != device) enable_peer_access(source, device);
            auto stream = cuda_side_stream(CUDA_COPY_STREAM, device);
            cuda_side_stream_wait_default(CUDA_COPY_STREAM, device, source);
            checkCudaErrors(cudaMemcpyAsync(dst, src, size, copy_kind(dst_device, src_device), stream));
            cuda_default_stream_wait_side(CUDA_COPY_STREAM, device, device);
            if (source != device)
                cuda_default_stream_wait_side(CUDA_COPY_STREAM, device, source);
            if (!ordered) checkCudaErrors(cudaStreamSynchronize(stream));
        } else {
            checkCudaErrors(cudaMemcpy(dst, src, size, copy_kind(dst_device, src_device)));
        }
    });
}
#endif

} // namespace jittor
