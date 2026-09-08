#pragma once
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "runtime/executor_entry.h"

namespace jittor {

// Cleanup must select the allocating CUDA device without changing Runtime
// placement or invoking device-switch hooks (which may create new handles).
struct CacheDeviceScope {
    int previous = -1;
    int device;
    bool active = false;
    explicit CacheDeviceScope(int target, bool reporting = false) : device(target) {
        auto status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != device) status = cudaSetDevice(device);
        if (reporting) {
            peekCudaErrorsAlways(status);
            active = status == cudaSuccess;
        } else {
            checkCudaErrors(status);
            active = true;
        }
    }
    ~CacheDeviceScope() {
        if (active && previous != device) peekCudaErrorsAlways(cudaSetDevice(previous));
    }
    CacheDeviceScope(const CacheDeviceScope&) = delete;
    CacheDeviceScope& operator=(const CacheDeviceScope&) = delete;
};

inline void wait_for_cached_plan(cudaStream_t stream) {
    // Called with ExecutorEntryScope during explicit cache operations. At
    // shutdown it keeps the GIL unchanged because there is no executor entry.
    cudaError_t status;
    {
        DeviceWaitScope wait;
        status = cudaStreamSynchronize(stream);
    }
    peekCudaErrorsAlways(status);
}

} // namespace jittor
