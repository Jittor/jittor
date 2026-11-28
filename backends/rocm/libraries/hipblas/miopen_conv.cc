#include "miopen_conv.h"
#include "hipblas_wrapper.h"
#include "convolution.h"
#include "core/executor.h"
#include "mem/allocator.h"
#include "runtime/backend.h"
#include "utils/log.h"
#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
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

void check_hip(hipError_t status, const char* operation) {
    CHECK(status == hipSuccess) << operation << ": " << hipGetErrorString(status);
}

bool unsupported(miopenStatus_t status) {
    return status == miopenStatusNotImplemented || status == miopenStatusUnsupportedOp;
}

miopenDataType_t tensor_type(NanoString dtype) {
    if (dtype == ns_float16) return miopenHalf;
    if (dtype == ns_bfloat16) return miopenBFloat16;
    return miopenFloat;
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

    void set_tensor(miopenTensorDescriptor_t& desc, const NanoVector& shape,
                    miopenDataType_t dtype) {
        check_miopen(miopenCreateTensorDescriptor(&desc), "miopenCreateTensorDescriptor");
        check_miopen(miopenSet4dTensorDescriptor(desc, dtype,
            shape[0], shape[1], shape[2], shape[3]), "miopenSet4dTensorDescriptor");
    }

    void init(const NanoVector& xs, const NanoVector& ws, const NanoVector& ys,
            miopenDataType_t dtype, const NanoVector& stride,
            const NanoVector& padding, const NanoVector& dilation) {
        set_tensor(x, xs, dtype);
        set_tensor(w, ws, dtype);
        set_tensor(y, ys, dtype);
        check_miopen(miopenCreateConvolutionDescriptor(&conv), "miopenCreateConvolutionDescriptor");
        check_miopen(miopenInitConvolutionDescriptor(conv, miopenConvolution,
            padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1]),
            "miopenInitConvolutionDescriptor");
    }
};

using Geometry = std::array<int64, 18>;

struct Plan {
    Descriptors descriptors;
    bool available = false;
    uint64_t solution = 0;
    size_t workspace = 0;
    bool method_tuned = false;
    bool use_gemm = false;
};

void find_plan(miopenHandle_t handle, Plan& plan, MiopenDirection direction) {
    const auto& d = plan.descriptors;
    auto first = d.w, second = d.x, output = d.y;
    auto count_fn = miopenConvolutionForwardGetSolutionCount;
    auto solution_fn = miopenConvolutionForwardGetSolution;
    if (direction == MiopenDirection::BackwardData) {
        first = d.y; second = d.w; output = d.x;
        count_fn = miopenConvolutionBackwardDataGetSolutionCount;
        solution_fn = miopenConvolutionBackwardDataGetSolution;
    } else if (direction == MiopenDirection::BackwardWeights) {
        first = d.y; second = d.x; output = d.w;
        count_fn = miopenConvolutionBackwardWeightsGetSolutionCount;
        solution_fn = miopenConvolutionBackwardWeightsGetSolution;
    }
    size_t count = 0;
    auto status = count_fn(handle, first, second, d.conv, output, &count);
    if (unsupported(status)) return;
    check_miopen(status, "miopenConvolutionGetSolutionCount");
    if (!count) return;
    std::vector<miopenConvSolution_t> solutions(count);
    size_t found = 0;
    status = solution_fn(handle, first, second, d.conv, output,
        count, &found, solutions.data());
    if (unsupported(status)) return;
    check_miopen(status, "miopenConvolutionGetSolution");
    CHECK(found <= count);
    for (size_t i=0; i<found; ++i) {
        const auto& solution = solutions[i];
        if (solution.workspace_size > workspace_limit) continue;
        plan.available = true;
        plan.solution = solution.solution_id;
        plan.workspace = solution.workspace_size;
        return;
    }
}

struct HandleState {
    miopenHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
};

struct State {
    std::mutex mutex;
    std::map<int, HandleState> handles;
    std::map<Geometry, unique_ptr<Plan>> plans;
    bool cleanup_registered = false;

    void shutdown() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        plans.clear();
        int previous = -1;
        bool restore = hipGetDevice(&previous) == hipSuccess;
        for (const auto& entry : handles) {
            if (!entry.second.handle) continue;
            if (hipSetDevice(entry.first) != hipSuccess) {
                std::fprintf(stderr, "MIOpen teardown: cannot select device %d\n", entry.first);
                continue;
            }
            auto status = miopenDestroy(entry.second.handle);
            if (status != miopenStatusSuccess)
                std::fprintf(stderr, "MIOpen teardown: %s\n", miopenGetErrorString(status));
        }
        handles.clear();
        cleanup_registered = false;
        if (restore && hipSetDevice(previous) != hipSuccess)
            std::fprintf(stderr, "MIOpen teardown: cannot restore device %d\n", previous);
    }
};

State& state() {
    static State instance;
    return instance;
}

void shutdown_miopen() { state().shutdown(); }

miopenHandle_t bind_handle(State& s, int device, hipStream_t stream) {
    if (!s.cleanup_registered) {
        register_cleanup_callback(&shutdown_miopen);
        s.cleanup_registered = true;
    }
    auto& entry = s.handles[device];
    if (!entry.handle) {
        check_miopen(miopenCreateWithStream(&entry.handle, stream), "miopenCreateWithStream");
        entry.stream = stream;
    } else if (entry.stream != stream) {
        check_miopen(miopenSetStream(entry.handle, stream), "miopenSetStream");
        entry.stream = stream;
    }
    return entry.handle;
}

Plan& plan_for(State& s, miopenHandle_t handle, int device, const NanoVector& x,
    const NanoVector& w, const NanoVector& y, NanoString dtype, MiopenDirection direction,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
    auto type = tensor_type(dtype);
    Geometry key = {{device, int(type), int(direction), x[0], x[1], x[2], x[3],
        w[0], w[2], w[3], y[2], y[3], stride[0], stride[1],
        padding[0], padding[1], dilation[0], dilation[1]}};
    auto found = s.plans.find(key);
    if (found != s.plans.end()) return *found->second;
    unique_ptr<Plan> plan(new Plan);
    plan->descriptors.init(x, w, y, type, stride, padding, dilation);
    find_plan(handle, *plan, direction);
    return *s.plans.emplace(key, move(plan)).first->second;
}

void launch_miopen(miopenHandle_t handle, const Plan& plan, MiopenDirection direction,
                   Var* x, Var* w, Var* y) {
    const auto& d = plan.descriptors;
    Allocation workspace(runtime_executor().temp_allocator, plan.workspace);
    if (direction == MiopenDirection::Forward)
        check_miopen(miopenConvolutionForwardImmediate(handle,
            d.w, w->mem_ptr, d.x, x->mem_ptr, d.conv, d.y, y->mem_ptr,
            workspace.ptr, plan.workspace, plan.solution), "miopenConvolutionForwardImmediate");
    else if (direction == MiopenDirection::BackwardData)
        check_miopen(miopenConvolutionBackwardDataImmediate(handle,
            d.y, y->mem_ptr, d.w, w->mem_ptr, d.conv, d.x, x->mem_ptr,
            workspace.ptr, plan.workspace, plan.solution), "miopenConvolutionBackwardDataImmediate");
    else
        check_miopen(miopenConvolutionBackwardWeightsImmediate(handle,
            d.y, y->mem_ptr, d.x, x->mem_ptr, d.conv, d.w, w->mem_ptr,
            workspace.ptr, plan.workspace, plan.solution), "miopenConvolutionBackwardWeightsImmediate");
}

bool pointwise(Var* w, const NanoVector& stride, const NanoVector& padding) {
    return w->shape[2] == 1 && w->shape[3] == 1 && stride[0] == 1 && stride[1] == 1
        && padding[0] == 0 && padding[1] == 0;
}

bool gemm_weights_supported(Var* x, Var* w, Var* y,
        const NanoVector& stride, const NanoVector& padding) {
    if (x->dtype() != ns_float32) return false;
    if (pointwise(w, stride, padding)) return true;
    int64 inner = w->num / w->shape[0], spatial = y->shape[2]*y->shape[3];
    return inner <= int64(workspace_limit/sizeof(float))/spatial;
}

// The one-batch columns layout is the existing native HIP convolution
// layout. Accumulate weight gradients directly, without a transposed copy
// of the activation or an expanded rank-7 tensor.
void launch_gemm_weights(Var* x, Var* w, Var* y, const NanoVector& stride,
    const NanoVector& padding, const NanoVector& dilation, hipStream_t stream) {
    int batch=x->shape[0], channels=x->shape[1], height=x->shape[2], width=x->shape[3];
    int oc=w->shape[0], kh=w->shape[2], kw=w->shape[3];
    int inner=channels*kh*kw, oh=y->shape[2], ow=y->shape[3], spatial=oh*ow;
    bool direct = pointwise(w, stride, padding);
    int64 count = int64(inner)*spatial;
    Allocation workspace(runtime_executor().temp_allocator, direct ? 0 : count*sizeof(float));
    auto handle = hipblas_bind_stream(w->device_id);
    const float alpha=1.f;
    for (int n=0; n<batch; ++n) {
        const float* input=x->ptr<float>() + int64(n)*channels*height*width;
        const float* columns=input;
        if (!direct) {
            auto* buffer=static_cast<float*>(workspace.ptr);
            check_hip(hip_conv_columns(input, buffer, count, channels, height, width,
                kh, kw, spatial, ow, 0, stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], stream), "HIP weight-gradient columns");
            columns=buffer;
        }
        const float beta = n ? 1.f : 0.f;
        check_hipblas(hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
            inner, oc, spatial, &alpha, columns, spatial,
            y->ptr<float>()+int64(n)*oc*spatial, spatial, &beta,
            w->ptr<float>(), inner), "hipblas weight gradient");
    }
}

bool gemm_forward_supported(Var* x, Var* w, Var* y,
        const NanoVector& stride, const NanoVector& padding) {
    if (pointwise(w, stride, padding)) return true;
    int64 inner=w->num/w->shape[0], spatial=y->shape[2]*y->shape[3];
    int bytes=x->dtype() == ns_float32 ? 4 : 2;
    return inner <= int64(workspace_limit/bytes)/spatial/x->shape[0];
}

void launch_gemm_forward(Var* x, Var* w, Var* y,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation,
    hipStream_t stream) {
    int batch=x->shape[0], channels=x->shape[1], height=x->shape[2], width=x->shape[3];
    int oc=w->shape[0], kh=w->shape[2], kw=w->shape[3];
    int inner=channels*kh*kw, oh=y->shape[2], ow=y->shape[3], spatial=oh*ow;
    int64 plane=int64(inner)*spatial;
    bool direct=pointwise(w,stride,padding), fp32=x->dtype()==ns_float32;
    int bytes=fp32 ? 4 : 2;
    Allocation workspace(runtime_executor().temp_allocator, direct ? 0 : plane*batch*bytes);
    const void* columns=x->mem_ptr;
    if (!direct) {
        auto status = fp32
            ? hip_conv_columns(x->ptr<float>(),static_cast<float*>(workspace.ptr),
                plane*batch,channels,height,width,kh,kw,spatial,ow,0,
                stride[0],stride[1],padding[0],padding[1],dilation[0],dilation[1],stream)
            : hip_conv_columns_half(x->mem_ptr,workspace.ptr,plane*batch,channels,height,width,
                kh,kw,spatial,ow,0,stride[0],stride[1],padding[0],padding[1],
                dilation[0],dilation[1],x->dtype()==ns_bfloat16,stream);
        check_hip(status,"HIP forward columns");
        columns=workspace.ptr;
    }
    auto handle=hipblas_bind_stream(y->device_id);
    const float alpha=1.f,beta=0.f;
    if (fp32) {
        if (batch == 1)
            check_hipblas(hipblasSgemm(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,spatial,oc,inner,
                &alpha,static_cast<const float*>(columns),spatial,w->ptr<float>(),inner,
                &beta,y->ptr<float>(),spatial),"hipblas convolution forward");
        else
            check_hipblas(hipblasSgemmStridedBatched(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,
                spatial,oc,inner,&alpha,static_cast<const float*>(columns),spatial,plane,
                w->ptr<float>(),inner,0,&beta,y->ptr<float>(),spatial,int64(oc)*spatial,batch),
                "hipblas batched convolution forward");
    } else {
        auto type=x->dtype()==ns_bfloat16 ? HIPBLAS_R_16B : HIPBLAS_R_16F;
        for (int n=0; n<batch; ++n)
            check_hipblas(hipblasGemmEx(handle,HIPBLAS_OP_N,HIPBLAS_OP_N,
                spatial,oc,inner,&alpha,static_cast<const char*>(columns)+int64(n)*plane*bytes,
                type,spatial,w->mem_ptr,type,inner,&beta,
                static_cast<char*>(y->mem_ptr)+int64(n)*oc*spatial*bytes,type,spatial,
                HIPBLAS_R_32F,HIPBLAS_GEMM_DEFAULT),"hipblas low precision convolution forward");
    }
}

struct Events {
    hipEvent_t start = nullptr, end = nullptr;
    ~Events() {
        if (start) hipEventDestroy(start);
        if (end) hipEventDestroy(end);
    }
};

void launch_alternative(MiopenDirection direction, Var* x, Var* w, Var* y,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation,
    hipStream_t stream) {
    if (direction == MiopenDirection::Forward)
        launch_gemm_forward(x,w,y,stride,padding,dilation,stream);
    else
        launch_gemm_weights(x,w,y,stride,padding,dilation,stream);
}

void choose_method(Plan& plan, miopenHandle_t handle, MiopenDirection direction,
    Var* x, Var* w, Var* y, const NanoVector& stride, const NanoVector& padding,
    const NanoVector& dilation, hipStream_t stream) {
    if (plan.method_tuned || direction == MiopenDirection::BackwardData) return;
    bool supported = direction == MiopenDirection::Forward
        ? gemm_forward_supported(x,w,y,stride,padding)
        : gemm_weights_supported(x,w,y,stride,padding);
    if (!supported) return;
    hipStreamCaptureStatus capture;
    check_hip(hipStreamIsCapturing(stream, &capture), "hipStreamIsCapturing");
    if (capture != hipStreamCaptureStatusNone) return;
    Events events;
    check_hip(hipEventCreate(&events.start), "hipEventCreate");
    check_hip(hipEventCreate(&events.end), "hipEventCreate");
    auto measure = [&](auto launch) {
        launch();
        check_hip(hipEventRecord(events.start, stream), "hipEventRecord");
        for (int i=0; i<3; ++i) launch();
        check_hip(hipEventRecord(events.end, stream), "hipEventRecord");
        check_hip(hipEventSynchronize(events.end), "hipEventSynchronize");
        float elapsed=0;
        check_hip(hipEventElapsedTime(&elapsed, events.start, events.end), "hipEventElapsedTime");
        return elapsed;
    };
    float native = measure([&] { launch_miopen(handle,plan,direction,x,w,y); });
    float gemm = measure([&] { launch_alternative(direction,x,w,y,stride,padding,dilation,stream); });
    // Keep the library convolution when performance is close.
    plan.use_gemm = gemm < native * .9f;
    plan.method_tuned = true;
}

} // namespace

void with_miopen_handle(int device, const std::function<void(miopenHandle_t)>& launch) {
    const auto& backend = backend_ops(BackendId::Rocm);
    ASSERTop(backend.current_device(), ==, device) << "MIOpen must run on its input device";
    auto stream = static_cast<hipStream_t>(backend.compute_stream(device));
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    launch(bind_handle(s, device, stream));
}

bool miopen_training_available(const NanoVector& x, const NanoVector& w,
    const NanoVector& y, NanoString dtype, int device,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
    const auto& backend = backend_ops(BackendId::Rocm);
    if (backend.current_device() != device) return false;
    auto stream = static_cast<hipStream_t>(backend.compute_stream(device));
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto handle = bind_handle(s, device, stream);
    for (auto direction : {MiopenDirection::Forward, MiopenDirection::BackwardData,
                           MiopenDirection::BackwardWeights})
        if (!plan_for(s, handle, device, x,w,y,dtype,direction,stride,padding,dilation).available)
            return false;
    return true;
}

bool try_miopen_conv(Var* x, Var* weight, Var* y, MiopenDirection direction,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
    int device = direction == MiopenDirection::Forward ? y->device_id
        : (direction == MiopenDirection::BackwardData ? x->device_id : weight->device_id);
    const auto& backend = backend_ops(BackendId::Rocm);
    ASSERTop(backend.current_device(), ==, device) << "MIOpen must run on its input device";
    auto stream = static_cast<hipStream_t>(backend.compute_stream(device));
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto handle = bind_handle(s, device, stream);
    auto& plan = plan_for(s, handle, device, x->shape,weight->shape,y->shape,
        x->dtype(),direction,stride,padding,dilation);
    if (!plan.available) return false;
    choose_method(plan,handle,direction,x,weight,y,stride,padding,dilation,stream);
    if (plan.use_gemm) {
        launch_alternative(direction,x,weight,y,stride,padding,dilation,stream);
        return true;
    }
    launch_miopen(handle,plan,direction,x,weight,y);
    return true;
}

bool try_miopen_conv_inference(Var* x, Var* weight, Var* y,
    const NanoVector& stride, const NanoVector& padding, const NanoVector& dilation) {
    return try_miopen_conv(x,weight,y,MiopenDirection::Forward,stride,padding,dilation);
}

} // namespace jittor
