#include "hipblas_linear_bias_op.h"
#include "hipblas_wrapper.h"
#include "linear_bias.h"
#include "core/var.h"
#include "core/executor.h"
#include "mem/allocator.h"
#include "ops/op_register.h"
#include "runtime/backend.h"
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#include <memory>

namespace jittor {
namespace {

constexpr size_t workspace_limit = 32 << 20;

struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algorithm{};
    size_t workspace = 0;
    bool available = false;

    ~Plan() {
        if (desc) hipblasLtMatmulDescDestroy(desc);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }
};

struct DeviceState {
    hipblasLtHandle_t handle = nullptr;
    std::map<std::tuple<int,int,int,int>, unique_ptr<Plan>> plans;
    ~DeviceState() {
        plans.clear();
        if (handle) hipblasLtDestroy(handle);
    }
};

struct States {
    std::map<int, unique_ptr<DeviceState>> devices;
    ~States() {
        int previous = -1;
        bool restore = hipGetDevice(&previous) == hipSuccess;
        for (auto& entry : devices) {
            if (hipSetDevice(entry.first) == hipSuccess) entry.second.reset();
        }
        if (restore) hipSetDevice(previous);
    }
};

DeviceState& state_for(int device) {
    static States states;
    auto& state = states.devices[device];
    if (!state) {
        state.reset(new DeviceState);
        check_hipblas(hipblasLtCreate(&state->handle), "hipblasLtCreate");
    }
    return *state;
}

bool unavailable(hipblasStatus_t status) {
    return status == HIPBLAS_STATUS_NOT_SUPPORTED || status == HIPBLAS_STATUS_ARCH_MISMATCH
        || status == HIPBLAS_STATUS_INVALID_VALUE;
}

Plan& plan_for(DeviceState& state, int m, int n, int k, hipDataType type, const void* bias) {
    auto key = std::make_tuple(m, n, k, int(type));
    auto found = state.plans.find(key);
    if (found != state.plans.end()) return *found->second;
    unique_ptr<Plan> plan(new Plan);
    check_hipblas(hipblasLtMatmulDescCreate(&plan->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "hipblasLtMatmulDescCreate");
    hipblasOperation_t trans_a = HIPBLAS_OP_T, trans_b = HIPBLAS_OP_N;
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    check_hipblas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSA,
        &trans_a, sizeof(trans_a)), "hipblasLt TRANSA");
    check_hipblas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSB,
        &trans_b, sizeof(trans_b)), "hipblasLt TRANSB");
    check_hipblas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue, sizeof(epilogue)), "hipblasLt epilogue");
    check_hipblas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        &type, sizeof(type)), "hipblasLt bias dtype");
    check_hipblas(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias, sizeof(bias)), "hipblasLt bias pointer");
    check_hipblas(hipblasLtMatrixLayoutCreate(&plan->a, type, k, n, k), "hipblasLt A layout");
    check_hipblas(hipblasLtMatrixLayoutCreate(&plan->b, type, k, m, k), "hipblasLt B layout");
    check_hipblas(hipblasLtMatrixLayoutCreate(&plan->c, type, n, m, n), "hipblasLt C layout");
    struct Preference {
        hipblasLtMatmulPreference_t value = nullptr;
        ~Preference() { if (value) hipblasLtMatmulPreferenceDestroy(value); }
    } pref;
    check_hipblas(hipblasLtMatmulPreferenceCreate(&pref.value), "hipblasLt preference");
    size_t limit = workspace_limit;
    check_hipblas(hipblasLtMatmulPreferenceSetAttribute(pref.value,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)), "hipblasLt workspace preference");
    hipblasLtMatmulHeuristicResult_t results[4];
    int count = 0;
    auto status = hipblasLtMatmulAlgoGetHeuristic(state.handle, plan->desc,
        plan->a, plan->b, plan->c, plan->c, pref.value, 4, results, &count);
    if (!unavailable(status)) {
        check_hipblas(status, "hipblasLtMatmulAlgoGetHeuristic");
        for (int i=0; i<count; ++i) {
            if (results[i].state != HIPBLAS_STATUS_SUCCESS || results[i].workspaceSize > limit) continue;
            plan->algorithm = results[i].algo;
            plan->workspace = results[i].workspaceSize;
            plan->available = true;
            break;
        }
    }
    return *state.plans.emplace(key, move(plan)).first->second;
}

} // namespace

HipblasLinearBiasOp::HipblasLinearBiasOp(Var* x, Var* weight, Var* bias)
    : x(x), weight(weight), bias(bias) {
    USER_CHECK(x->dtype() == weight->dtype() && x->dtype() == bias->dtype()
        && (x->dtype() == ns_float16 || x->dtype() == ns_bfloat16))
        << "hipBLAS biased linear requires matching FP16 or BF16 inputs";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    x->set_flag(VarFlags::_needed_by_backward);
    weight->set_flag(VarFlags::_needed_by_backward);
    y = create_output(nullptr, x->dtype());
}

void HipblasLinearBiasOp::infer_shape() {
    USER_CHECK(x->shape.size() >= 2 && weight->shape.size() == 2 && bias->shape.size() == 1
        && x->shape[x->shape.size()-1] == weight->shape[1] && bias->shape[0] == weight->shape[0])
        << "invalid hipBLAS biased linear dimensions";
    NanoVector out;
    for (uint i=0; i+1<x->shape.size(); ++i) out.push_back(x->shape[i]);
    out.push_back(weight->shape[0]);
    y->set_shape(out);
}

void HipblasLinearBiasOp::run() {
    if (!y->num) return;
    int64 n64 = weight->shape[0], k64 = weight->shape[1], m64 = y->num/n64;
    USER_CHECK(n64 > 0 && k64 > 0 && m64 > 0 && n64 <= INT_MAX && k64 <= INT_MAX && m64 <= INT_MAX)
        << "hipBLAS biased linear dimensions must fit positive int32";
    int m = int(m64), n = int(n64), k = int(k64), device = y->device_id;
    auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(device));
    bool bf16 = x->dtype() == ns_bfloat16;
    auto type = bf16 ? HIP_R_16BF : HIP_R_16F;
    auto& state = state_for(device);
    const void* bias_ptr = bias->mem_ptr;
    auto& plan = plan_for(state, m, n, k, type, bias_ptr);
    const float alpha = 1.f, beta = 0.f;
    if (plan.available) {
        check_hipblas(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr, sizeof(bias_ptr)), "hipblasLt bias pointer");
        Allocation workspace(runtime_executor().temp_allocator, plan.workspace);
        auto status = hipblasLtMatmul(state.handle, plan.desc, &alpha,
            weight->mem_ptr, plan.a, x->mem_ptr, plan.b, &beta,
            y->mem_ptr, plan.c, y->mem_ptr, plan.c, &plan.algorithm,
            workspace.ptr, plan.workspace, stream);
        if (!unavailable(status)) {
            check_hipblas(status, "hipblasLtMatmul biased linear");
            return;
        }
    }
    // The SDK can decline a geometry. Retain the FP32 bias accumulation
    // semantics through ordinary hipBLAS plus a typed HIP epilogue.
    Allocation accumulated(runtime_executor().temp_allocator, y->num*sizeof(float));
    auto input_type = bf16 ? HIPBLAS_R_16B : HIPBLAS_R_16F;
    auto handle = hipblas_bind_stream(device);
    check_hipblas(hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, k,
        &alpha, weight->mem_ptr, input_type, k, x->mem_ptr, input_type, k,
        &beta, accumulated.ptr, HIPBLAS_R_32F, n, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT),
        "hipBLAS FP32 biased linear accumulation");
    auto status = hip_linear_bias_cast(static_cast<const float*>(accumulated.ptr),
        bias_ptr, y->mem_ptr, y->num, n, bf16, stream);
    CHECK(status == hipSuccess) << hipGetErrorString(status);
}

VarPtr HipblasLinearBiasOp::grad(Var*, Var* dout, Var*, int v_index) {
    auto matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("hipblas_matmul");
    auto cast = op_constructor<VarPtr, Var*, NanoString>("unary");
    VarPtr narrow;
    if (dout->dtype() != x->dtype()) {
        narrow = cast(dout, x->dtype());
        dout = narrow.ptr;
    }
    if (v_index == 0)
        return matmul(dout, weight, false, false);
    if (v_index == 1) {
        if (x->shape.size() == 2) return matmul(dout, x, true, false);
        auto reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");
        auto xx = reshape(x, {x->num/weight->shape[1], weight->shape[1]});
        auto dd = reshape(dout, {dout->num/weight->shape[0], weight->shape[0]});
        return matmul(dd, xx, true, false);
    }
    NanoVector dims;
    for (uint i=0; i+1<dout->shape.size(); ++i) dims.push_back(i);
    auto reduce = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");
    auto wide = cast(dout, ns_float32);
    return cast(reduce(wide, ns_add, dims, false), bias->dtype());
}

} // namespace jittor
