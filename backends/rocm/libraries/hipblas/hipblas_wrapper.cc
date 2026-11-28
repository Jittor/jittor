#include "hipblas_wrapper.h"
#include "runtime/backend.h"
#include "runtime/float32_precision.h"
#include <cstdio>
#include <map>
#include <tuple>

namespace jittor {
namespace {
struct HandleState {
    hipblasHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
    bool stream_bound = false;
    hipblasMath_t requested_math = HIPBLAS_DEFAULT_MATH;
    bool math_bound = false;
    std::map<std::tuple<int,int,int,int,int,int>, hipblasMath_t> inference_math;
};

struct Handles {
    std::map<int, HandleState> devices;
    std::map<int, bool> fp64;
    ~Handles() { hipblas_shutdown(); }
};

Handles& handles() {
    static Handles state;
    return state;
}
hipblasHandle_t bind_stream_math(int device, hipblasMath_t mode) {
    const auto& backend = backend_ops(BackendId::Rocm);
    // `device` comes from the op's own placement, not from the caller.
    ASSERTop(backend.current_device(), ==, device)
        << "hipBLAS must run on its input device";
    auto& state = handles().devices[device];
    if (!state.handle) {
        check_hipblas(hipblasCreate(&state.handle), "hipblasCreate");
        check_hipblas(hipblasSetPointerMode(state.handle, HIPBLAS_POINTER_MODE_HOST),
                     "hipblasSetPointerMode");
    }
    auto stream = static_cast<hipStream_t>(backend.compute_stream(device));
    if (!state.stream_bound || state.stream != stream) {
        check_hipblas(hipblasSetStream(state.handle, stream), "hipblasSetStream");
        state.stream = stream;
        state.stream_bound = true;
    }
    // Keep the handle's precision attached to each operation (including a
    // convolution after an XF32 GEMM), without reconfiguring identical calls.
    if (!state.math_bound || state.requested_math != mode) {
        auto status = hipblasSetMathMode(state.handle, mode);
        if (mode != HIPBLAS_DEFAULT_MATH
                && (status == HIPBLAS_STATUS_NOT_SUPPORTED || status == HIPBLAS_STATUS_ARCH_MISMATCH))
            status = hipblasSetMathMode(state.handle, HIPBLAS_DEFAULT_MATH);
        check_hipblas(status, "hipblasSetMathMode");
        state.requested_math = mode;
        state.math_bound = true;
    }
    return state.handle;
}

void check_hip(hipError_t status, const char* operation) {
    CHECK(status == hipSuccess) << operation << ": " << hipGetErrorString(status);
}

struct TimingEvents {
    hipEvent_t start = nullptr, end = nullptr;
    ~TimingEvents() {
        if (start) hipEventDestroy(start);
        if (end) hipEventDestroy(end);
    }
};
} // namespace

hipblasHandle_t hipblas_bind_stream(int device) {
    return bind_stream_math(device, HIPBLAS_DEFAULT_MATH);
}

void hipblas_sgemm_policy(int device, hipblasOperation_t trans_a, hipblasOperation_t trans_b,
    int m, int n, int k, const float* a, int lda, const float* b, int ldb,
    float* c, int ldc, bool inference, int64 stride_a, int64 stride_b, int64 stride_c,
    int batch_count) {
    const float alpha = 1, beta = 0;
    auto launch = [&](hipblasHandle_t handle) {
        if (batch_count)
            check_hipblas(hipblasSgemmStridedBatched(handle, trans_a, trans_b,
                m, n, k, &alpha, a, lda, stride_a, b, ldb, stride_b,
                &beta, c, ldc, stride_c, batch_count), "hipblasSgemmStridedBatched");
        else
            check_hipblas(hipblasSgemm(handle, trans_a, trans_b, m, n, k,
                &alpha, a, lda, b, ldb, &beta, c, ldc), "hipblasSgemm");
    };
    auto mode = float32_matmul_tier() >= F32_HIGH
        ? HIPBLAS_XF32_XDL_MATH : HIPBLAS_DEFAULT_MATH;
    // Relaxed precision permits full FP32 as well. Some ROCm XF32
    // solutions are slower than SGEMM for the same geometry. Measure once
    // for substantial inference products, outside any device recording.
    if (mode == HIPBLAS_XF32_XDL_MATH && inference
            && double(m)*n*k*(batch_count ? batch_count : 1) >= (1LL << 24)) {
        auto& state = handles().devices[device];
        auto key = std::make_tuple(m, n, k, int(trans_a), int(trans_b), batch_count);
        auto found = state.inference_math.find(key);
        if (found != state.inference_math.end()) {
            mode = found->second;
        } else {
            bind_stream_math(device, mode);
            hipStreamCaptureStatus capture;
            check_hip(hipStreamIsCapturing(state.stream, &capture), "hipStreamIsCapturing");
            if (capture == hipStreamCaptureStatusNone) {
                TimingEvents events;
                check_hip(hipEventCreate(&events.start), "hipEventCreate");
                check_hip(hipEventCreate(&events.end), "hipEventCreate");
                auto measure = [&](hipblasMath_t math) {
                    auto handle = bind_stream_math(device, math);
                    launch(handle);
                    check_hip(hipEventRecord(events.start, state.stream), "hipEventRecord");
                    for (int i = 0; i < 3; ++i) launch(handle);
                    check_hip(hipEventRecord(events.end, state.stream), "hipEventRecord");
                    check_hip(hipEventSynchronize(events.end), "hipEventSynchronize");
                    float elapsed = 0;
                    check_hip(hipEventElapsedTime(&elapsed, events.start, events.end), "hipEventElapsedTime");
                    return elapsed;
                };
                float strict = measure(HIPBLAS_DEFAULT_MATH);
                float relaxed = measure(HIPBLAS_XF32_XDL_MATH);
                // Keep XF32 when timings are similar; a sizeable win is
                // required to prefer the higher-precision implementation.
                if (strict < relaxed * 0.9f) mode = HIPBLAS_DEFAULT_MATH;
                state.inference_math.emplace(key, mode);
            }
        }
    }
    launch(bind_stream_math(device, mode));
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

void hipblas_shutdown() noexcept {
    auto& devices = handles().devices;
    if (devices.empty()) return;
    int previous = -1;
    const bool restore = hipGetDevice(&previous) == hipSuccess;
    for (auto& entry : devices) {
        if (!entry.second.handle) continue;
        const auto selected = hipSetDevice(entry.first);
        if (selected != hipSuccess) {
            std::fprintf(stderr, "hipBLAS teardown: hipSetDevice(%d) failed: %d\n",
                         entry.first, int(selected));
            continue;
        }
        const auto status = hipblasDestroy(entry.second.handle);
        if (status != HIPBLAS_STATUS_SUCCESS)
            std::fprintf(stderr, "hipBLAS teardown: hipblasDestroy failed: %d\n", int(status));
        entry.second.handle = nullptr;
    }
    devices.clear();
    if (restore && hipSetDevice(previous) != hipSuccess)
        std::fprintf(stderr, "hipBLAS teardown: cannot restore device %d\n", previous);
}

} // namespace jittor
