#include "hipblas_wrapper.h"
#include "runtime/backend.h"
#include "runtime/float32_precision.h"
#include <cstdio>
#include <map>

namespace jittor {
namespace {
struct Handles {
    std::map<int, hipblasHandle_t> devices;
    std::map<int, bool> fp64;
    ~Handles() { hipblas_shutdown(); }
};

Handles& handles() {
    static Handles state;
    return state;
}
} // namespace

hipblasHandle_t hipblas_bind_stream(int device) {
    const auto& backend = backend_ops(BackendId::Rocm);
    // `device` comes from the op's own placement, not from the caller.
    ASSERTop(backend.current_device(), ==, device)
        << "hipBLAS must run on its input device";
    auto& handle = handles().devices[device];
    if (!handle) {
        hipblasHandle_t created = nullptr;
        check_hipblas(hipblasCreate(&created), "hipblasCreate");
        handle = created;
    }
    check_hipblas(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
                 "hipblasSetPointerMode");
    check_hipblas(hipblasSetStream(handle,
        static_cast<hipStream_t>(backend.compute_stream(device))), "hipblasSetStream");
    return handle;
}

void hipblas_require_fp64(int device) {
    if (device < 0) device = backend_ops(BackendId::Rocm).current_device();
    auto& supported = handles().fp64;
    auto entry = supported.find(device);
    if (entry == supported.end()) {
        hipDeviceProp_t properties{};
        const auto status = hipGetDeviceProperties(&properties, device);
        CHECK(status == hipSuccess) << "hipGetDeviceProperties: " << hipGetErrorString(status);
        entry = supported.emplace(device, bool(properties.arch.hasDoubles)).first;
    }
    USER_CHECK(entry->second) << "ROCm device does not support true float64 computation";
}

void hipblas_matmul_math(hipblasHandle_t handle, bool float32) {
    // The shared handle must honor each captured operation's precision policy.
    // XF32 is the rocBLAS spelling of TF32. On devices without that mode,
    // retaining full FP32 accuracy is valid for a relaxed precision request.
    auto mode = float32 && float32_matmul_tier() >= F32_HIGH
        ? HIPBLAS_XF32_XDL_MATH : HIPBLAS_DEFAULT_MATH;
    auto status = hipblasSetMathMode(handle, mode);
    if (mode != HIPBLAS_DEFAULT_MATH
            && (status == HIPBLAS_STATUS_NOT_SUPPORTED || status == HIPBLAS_STATUS_ARCH_MISMATCH))
        status = hipblasSetMathMode(handle, HIPBLAS_DEFAULT_MATH);
    check_hipblas(status, "hipblasSetMathMode");
}

void hipblas_shutdown() noexcept {
    auto& devices = handles().devices;
    if (devices.empty()) return;
    int previous = -1;
    const bool restore = hipGetDevice(&previous) == hipSuccess;
    for (auto& entry : devices) {
        if (!entry.second) continue;
        const auto selected = hipSetDevice(entry.first);
        if (selected != hipSuccess) {
            std::fprintf(stderr, "hipBLAS teardown: hipSetDevice(%d) failed: %d\n",
                         entry.first, int(selected));
            continue;
        }
        const auto status = hipblasDestroy(entry.second);
        if (status != HIPBLAS_STATUS_SUCCESS)
            std::fprintf(stderr, "hipBLAS teardown: hipblasDestroy failed: %d\n", int(status));
        entry.second = nullptr;
    }
    devices.clear();
    if (restore && hipSetDevice(previous) != hipSuccess)
        std::fprintf(stderr, "hipBLAS teardown: cannot restore device %d\n", previous);
}

} // namespace jittor
