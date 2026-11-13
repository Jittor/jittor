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
               && (a->shape.size() == 2 || (!trans_a && a->dtype() == ns_float32)))
        << "hipBLAS matmul requires rank-2 inputs or an untransposed FP32 left tensor";
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
