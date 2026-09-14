#include "hipblas_wrapper.h"
#include "runtime/backend.h"
#include <cstdio>
#include <map>

namespace jittor {
namespace {
struct Handles {
    std::map<int, hipblasHandle_t> devices;
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
