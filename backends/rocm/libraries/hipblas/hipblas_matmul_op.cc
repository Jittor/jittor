#include "hipblas_matmul_op.h"
#include "hipblas_wrapper.h"
#include "gemm_layout.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/backend.h"

namespace jittor {
namespace {
int64 dense_rows(Var* a, Var* b, bool trans_a) {
    USER_CHECK(a->shape.size() >= 2 && b->shape.size() == 2
               && (a->shape.size() == 2 || (!trans_a && (a->dtype() == ns_float32 || a->dtype() == ns_float16
                   || a->dtype() == ns_bfloat16))))
        << "hipBLAS matmul requires rank-2 inputs or an untransposed FP32/FP16/BF16 left tensor";
    int64 rows = 1;
    for (int i = 0; i + 1 < a->shape.size(); ++i) {
        const int64 size = a->shape[i];
        USER_CHECK(size >= 0 && (size == 0 || rows <= INT_MAX / size))
            << "hipBLAS flattened rows must fit nonnegative int32";
        rows *= size;
    }
    return rows;
}
} // namespace


HipblasMatmulOp::HipblasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : HipblasMatmulOp(a, b, trans_a, trans_b, false) {}

HipblasMatmulOp::HipblasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b, bool fp32_output)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    USER_CHECK(a->dtype() == b->dtype()
               && (a->dtype() == ns_float32 || a->dtype() == ns_float64
                   || a->dtype() == ns_float16 || a->dtype() == ns_bfloat16
                   || a->dtype() == ns_int8))
        << "hipBLAS matmul requires matching float32, float64, float16, bfloat16 or int8 inputs";
    if (a->dtype() == ns_float64) hipblas_require_fp64(a->device_id);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    USER_CHECK(!fp32_output || a->dtype() == ns_float16 || a->dtype() == ns_bfloat16)
        << "hipBLAS FP32 accumulator output requires FP16 or BF16 operands";
    c = create_output(nullptr, fp32_output ? ns_float32
        : (a->dtype() == ns_int8 ? ns_int32 : a->dtype()));
}

void HipblasMatmulOp::infer_shape() {
    const int64 rows = dense_rows(a, b, trans_a);
    const auto layout = hipblas_gemm_layout(
        rows, a->shape[a->shape.size()-1],
        b->shape[0], b->shape[1], trans_a, trans_b);
    NanoVector shape;
    if (a->shape.size() > 2) {
        for (int i = 0; i + 1 < a->shape.size(); ++i) shape.push_back(a->shape[i]);
    } else shape.push_back(layout.rows);
    shape.push_back(layout.columns);
    c->set_shape(shape);
}

void HipblasMatmulOp::run() {
    if (!c->num) return;
    const int64 rows = dense_rows(a, b, trans_a);
    const auto layout = hipblas_gemm_layout(
        rows, a->shape[a->shape.size()-1],
        b->shape[0], b->shape[1], trans_a, trans_b);
    const int device = c->device_id;
    if (!layout.inner) {
        auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(device));
        const auto status = hipMemsetAsync(c->mem_ptr, 0, c->size, stream);
        // hipMemsetAsync failing is a runtime fault, not caller input.
        CHECK(status == hipSuccess) << "hipBLAS empty matmul: " << hipGetErrorString(status);
        return;
    }
    const auto left = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const auto right = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    if (a->dtype() == ns_float32) {
        hipblas_sgemm_policy(device, left, right, layout.columns, layout.rows, layout.inner,
            b->ptr<float>(), layout.lda, a->ptr<float>(), layout.ldb,
            c->ptr<float>(), layout.ldc, c->is_stop_grad());
        return;
    }
    auto handle = hipblas_bind_stream(device);
    if (a->dtype() == ns_float64) {
        hipblas_require_fp64(device);
        const double alpha = 1, beta = 0;
        check_hipblas(hipblasDgemm(handle, left, right,
            layout.columns, layout.rows, layout.inner, &alpha,
            b->ptr<double>(), layout.lda, a->ptr<double>(), layout.ldb,
            &beta, c->ptr<double>(), layout.ldc), "hipblasDgemm");
    } else if (a->dtype() == ns_int8) {
        const int32 alpha = 1, beta = 0;
        check_hipblas(hipblasGemmEx(handle, left, right,
            layout.columns, layout.rows, layout.inner, &alpha,
            b->mem_ptr, HIPBLAS_R_8I, layout.lda, a->mem_ptr, HIPBLAS_R_8I, layout.ldb,
            &beta, c->mem_ptr, HIPBLAS_R_32I, layout.ldc, HIPBLAS_R_32I,
            HIPBLAS_GEMM_DEFAULT), "hipblasGemmEx int8");
    } else {
        const float alpha = 1, beta = 0;
        const auto type = a->dtype() == ns_float16 ? HIPBLAS_R_16F : HIPBLAS_R_16B;
        check_hipblas(hipblasGemmEx(handle, left, right,
            layout.columns, layout.rows, layout.inner, &alpha,
            b->mem_ptr, type, layout.lda, a->mem_ptr, type, layout.ldb,
            &beta, c->mem_ptr, c->dtype() == ns_float32 ? HIPBLAS_R_32F : type,
            layout.ldc, HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT), "hipblasGemmEx half");
    }
}

VarPtr HipblasMatmulOp::grad(Var*, Var* dout, Var*, int v_index) {
    // A FP32 consumer (e.g. autocast LayerNorm or cross entropy) can
    // supply a wider upstream gradient than the saved half GEMM operands.
    VarPtr cast_dout;
    if (dout->dtype() != a->dtype()) {
        auto cast = op_constructor<VarPtr, Var*, NanoString>("unary");
        cast_dout = cast(dout, a->dtype());
        dout = cast_dout.ptr;
    }
    auto matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("hipblas_matmul");
    if (v_index == 0)
        return trans_a ? matmul(b, dout, trans_b, true)
                       : matmul(dout, b, false, !trans_b);
    if (a->shape.size() > 2) {
        // The weight gradient contracts every leading input dimension. Keep
        // these reshape edges in backward so higher derivatives retain both
        // original operands; inference needs neither view node.
        auto reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");
        const int64 rows = dense_rows(a, b, false);
        VarPtr aa = reshape(a, {rows, a->shape[a->shape.size()-1]});
        VarPtr dd = reshape(dout, {rows, dout->shape[dout->shape.size()-1]});
        return trans_b ? matmul(dd, aa, true, false)
                       : matmul(aa, dd, true, false);
    }
    return trans_b ? matmul(dout, a, true, trans_a)
                   : matmul(a, dout, !trans_a, false);
}

} // namespace jittor
