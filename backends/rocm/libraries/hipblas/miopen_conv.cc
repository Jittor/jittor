#include "miopen_conv.h"
#include "core/executor.h"
#include "mem/allocator.h"
#include "runtime/backend.h"
#include "utils/log.h"
#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>
#include <array>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace jittor {
EXTERN_LIB void register_cleanup_callback(void (*cb)());

namespace {

constexpr size_t workspace_limit = 256ULL << 20;

void check_miopen(miopenStatus_t status, const char* operation) {
    CHECK(status == miopenStatusSuccess)
        << operation << " failed: " << miopenGetErrorString(status);
}

bool unsupported(miopenStatus_t status) {
    return status == miopenStatusNotImplemented || status == miopenStatusUnsupportedOp;
}

struct Descriptors {
    miopenTensorDescriptor_t x = nullptr, w = nullptr, y = nullptr;
    miopenConvolutionDescriptor_t conv = nullptr;

    ~Descriptors() {
        if (conv) miopenDestroyConvolutionDescriptor(conv);
        if (y) miopenDestroyTensorDescriptor(y);
        if (w) miopenDestroyTensorDescriptor(w);
        if (x) miopenDestroyTensorDescriptor(x);
    }

    void set_tensor(miopenTensorDescriptor_t& desc, Var* var) {
        check_miopen(miopenCreateTensorDescriptor(&desc), "miopenCreateTensorDescriptor");
        check_miopen(miopenSet4dTensorDescriptor(desc, miopenFloat,
            var->shape[0], var->shape[1], var->shape[2], var->shape[3]),
            "miopenSet4dTensorDescriptor");
    }

    void init(Var* input, Var* weight, Var* output,
            const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
        set_tensor(x, input);
        set_tensor(w, weight);
        set_tensor(y, output);
        check_miopen(miopenCreateConvolutionDescriptor(&conv), "miopenCreateConvolutionDescriptor");
        check_miopen(miopenInitConvolutionDescriptor(conv, miopenConvolution,
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1]),
            "miopenInitConvolutionDescriptor");
    }
};

using Geometry = std::array<int64, 16>;

struct Plan {
    bool available = false;
    uint64_t solution = 0;
    size_t workspace = 0;
};

Plan find_plan(miopenHandle_t handle, const Descriptors& d) {
    size_t count = 0;
    auto status = miopenConvolutionForwardGetSolutionCount(handle, d.w, d.x, d.conv, d.y, &count);
    if (unsupported(status)) return {};
    check_miopen(status, "miopenConvolutionForwardGetSolutionCount");
    if (!count) return {};

    std::vector<miopenConvSolution_t> solutions(count);
    size_t found = 0;
    status = miopenConvolutionForwardGetSolution(handle, d.w, d.x, d.conv, d.y,
        count, &found, solutions.data());
    if (unsupported(status)) return {};
    check_miopen(status, "miopenConvolutionForwardGetSolution");
    CHECK(found <= count);
    for (size_t i = 0; i < found; ++i) {
        const auto& solution = solutions[i];
        if (solution.workspace_size > workspace_limit) continue;
        Plan plan;
        plan.available = true;
        plan.solution = solution.solution_id;
        plan.workspace = solution.workspace_size;
        LOGv << "MIOpen convolution solution " << plan.solution
             << " workspace " << plan.workspace;
        return plan;
    }
    return {};
}

struct State {
    std::mutex mutex;
    std::map<int, miopenHandle_t> handles;
    std::map<Geometry, Plan> plans;
    bool cleanup_registered = false;

    void shutdown() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (handles.empty()) {
            cleanup_registered = false;
            return;
        }
        int previous = -1;
        const bool restore = hipGetDevice(&previous) == hipSuccess;
        for (const auto& entry : handles) {
            if (!entry.second) continue;
            if (hipSetDevice(entry.first) != hipSuccess) {
                std::fprintf(stderr, "MIOpen teardown: cannot select device %d\n", entry.first);
                continue;
            }
            const auto status = miopenDestroy(entry.second);
            if (status != miopenStatusSuccess)
                std::fprintf(stderr, "MIOpen teardown: %s\n", miopenGetErrorString(status));
        }
        handles.clear();
        plans.clear();
        cleanup_registered = false;
        if (restore && hipSetDevice(previous) != hipSuccess)
            std::fprintf(stderr, "MIOpen teardown: cannot restore device %d\n", previous);
    }
};

State& state() {
    static State instance;
    return instance;
}

void shutdown_miopen() {
    state().shutdown();
}

} // namespace

bool try_miopen_conv_inference(Var* x, Var* weight, Var* y,
        const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
    const int device = y->device_id;
    const auto& backend = backend_ops(BackendId::Rocm);
    ASSERTop(backend.current_device(), ==, device) << "MIOpen must run on its input device";

    Geometry key = {{device, x->shape[0], x->shape[1], x->shape[2], x->shape[3],
        weight->shape[0], weight->shape[2], weight->shape[3], y->shape[2], y->shape[3],
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1]}};
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    // SDK handles must be released while the runtime is still alive.
    if (!s.cleanup_registered) {
        register_cleanup_callback(&shutdown_miopen);
        s.cleanup_registered = true;
    }
    auto stream = static_cast<hipStream_t>(backend.compute_stream(device));
    auto& handle = s.handles[device];
    if (!handle) {
        miopenHandle_t created = nullptr;
        check_miopen(miopenCreateWithStream(&created, stream), "miopenCreateWithStream");
        handle = created;
    } else {
        check_miopen(miopenSetStream(handle, stream), "miopenSetStream");
    }

    Descriptors descriptors;
    descriptors.init(x, weight, y, stride, padding, dilation);
    auto it = s.plans.find(key);
    if (it == s.plans.end())
        it = s.plans.emplace(key, find_plan(handle, descriptors)).first;
    const Plan& plan = it->second;
    if (!plan.available) return false;

    Allocation workspace(runtime_executor().temp_allocator, plan.workspace);
    check_miopen(miopenConvolutionForwardImmediate(handle,
        descriptors.w, weight->ptr<float>(), descriptors.x, x->ptr<float>(),
        descriptors.conv, descriptors.y, y->ptr<float>(), workspace.ptr, plan.workspace,
        plan.solution), "miopenConvolutionForwardImmediate");
    return true;
}

} // namespace jittor
