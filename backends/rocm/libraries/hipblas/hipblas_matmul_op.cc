#include "hipblas_matmul_op.h"
#include "hipblas_wrapper.h"
#include "gemm_layout.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/backend.h"

namespace jittor {

HipblasMatmulOp::HipblasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    USER_CHECK(a->dtype() == b->dtype()
               && (a->dtype() == ns_float32 || a->dtype() == ns_float64))
        << "hipBLAS matmul requires matching float32 or float64 inputs";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    c = create_output(nullptr, a->dtype());
}

void HipblasMatmulOp::infer_shape() {
    USER_CHECK(a->shape.size() == 2 && b->shape.size() == 2)
        << "hipBLAS matmul requires rank-2 inputs";
    const auto layout = hipblas_gemm_layout(
        a->shape[0], a->shape[1], b->shape[0], b->shape[1], trans_a, trans_b);
    c->set_shape({layout.rows, layout.columns});
}

void HipblasMatmulOp::run() {
    if (!c->num) return;
    const auto layout = hipblas_gemm_layout(
        a->shape[0], a->shape[1], b->shape[0], b->shape[1], trans_a, trans_b);
    const int device = c->device_id;
    if (!layout.inner) {
        auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(device));
        const auto status = hipMemsetAsync(c->mem_ptr, 0, c->size, stream);
        USER_CHECK(status == hipSuccess) << "hipBLAS empty matmul: " << hipGetErrorString(status);
        return;
    }
    auto handle = hipblas_bind_stream(device);
    const auto left = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const auto right = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    if (a->dtype() == ns_float32) {
        const float alpha = 1, beta = 0;
        check_hipblas(hipblasSgemm(handle, left, right,
            layout.columns, layout.rows, layout.inner, &alpha,
            b->ptr<float>(), layout.lda, a->ptr<float>(), layout.ldb,
            &beta, c->ptr<float>(), layout.ldc), "hipblasSgemm");
    } else {
        const double alpha = 1, beta = 0;
        check_hipblas(hipblasDgemm(handle, left, right,
            layout.columns, layout.rows, layout.inner, &alpha,
            b->ptr<double>(), layout.lda, a->ptr<double>(), layout.ldb,
            &beta, c->ptr<double>(), layout.ldc), "hipblasDgemm");
    }
}

VarPtr HipblasMatmulOp::grad(Var*, Var* dout, Var*, int v_index) {
    auto matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("hipblas_matmul");
    if (v_index == 0)
        return trans_a ? matmul(b, dout, trans_b, true)
                       : matmul(dout, b, false, !trans_b);
    return trans_b ? matmul(dout, a, true, trans_a)
                   : matmul(a, dout, !trans_a, false);
}

} // namespace jittor
