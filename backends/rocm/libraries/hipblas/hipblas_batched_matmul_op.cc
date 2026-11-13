#include "hipblas_batched_matmul_op.h"
#include "hipblas_wrapper.h"
#include "gemm_layout.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/backend.h"

namespace jittor {

HipblasBatchedMatmulOp::HipblasBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    USER_CHECK(a->dtype() == ns_float32 && b->dtype() == ns_float32)
        << "hipBLAS batched matmul requires matching float32 inputs";
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    c = create_output(nullptr, a->dtype());
}

void HipblasBatchedMatmulOp::infer_shape() {
    const auto rank = a->shape.size();
    USER_CHECK(rank >= 3 && b->shape.size() == rank)
        << "hipBLAS batched matmul requires equal input ranks >= 3";
    for (uint axis=0; axis<rank-2; ++axis)
        USER_CHECK(a->shape[axis] == b->shape[axis])
            << "hipBLAS batched matmul batch dimensions must match at axis " << axis;
    const auto layout = hipblas_gemm_layout(
        a->shape[rank-2], a->shape[rank-1], b->shape[rank-2], b->shape[rank-1],
        trans_a, trans_b);
    NanoVector shape;
    for (uint axis=0; axis<rank-2; ++axis)
        shape.push_back(a->shape[axis]);
    shape.push_back(layout.rows);
    shape.push_back(layout.columns);
    c->set_shape(shape);
    batch_count = 0;
    if (!c->num) return;
    int64 count = 1;
    for (uint axis=0; axis<rank-2; ++axis) {
        USER_CHECK(a->shape[axis] > 0 && count <= INT_MAX / a->shape[axis])
            << "hipBLAS batched matmul batch count must fit int32";
        count *= a->shape[axis];
    }
    batch_count = int(count);
}

void HipblasBatchedMatmulOp::run() {
    if (!c->num) return;
    const auto rank = a->shape.size();
    const auto layout = hipblas_gemm_layout(
        a->shape[rank-2], a->shape[rank-1], b->shape[rank-2], b->shape[rank-1],
        trans_a, trans_b);
    const int device = c->device_id;
    if (!layout.inner) {
        auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(device));
        const auto status = hipMemsetAsync(c->mem_ptr, 0, c->size, stream);
        CHECK(status == hipSuccess) << "hipBLAS empty batched matmul: " << hipGetErrorString(status);
        return;
    }
    auto handle = hipblas_bind_stream(device);
    const auto left = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const auto right = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const int64 stride_a = int64(layout.rows) * layout.inner;
    const int64 stride_b = int64(layout.inner) * layout.columns;
    const int64 stride_c = int64(layout.rows) * layout.columns;
    // Row-major C=A*B is column-major C^T=B^T*A^T for every batch.
    const float alpha = 1, beta = 0;
    check_hipblas(hipblasSgemmStridedBatched(handle, left, right,
        layout.columns, layout.rows, layout.inner, &alpha,
        b->ptr<float>(), layout.lda, stride_b,
        a->ptr<float>(), layout.ldb, stride_a, &beta,
        c->ptr<float>(), layout.ldc, stride_c, batch_count), "hipblasSgemmStridedBatched");
}

VarPtr HipblasBatchedMatmulOp::grad(Var*, Var* dout, Var*, int v_index) {
    auto matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("hipblas_batched_matmul");
    if (v_index == 0)
        return trans_a ? matmul(b, dout, trans_b, true)
                       : matmul(dout, b, false, !trans_b);
    return trans_b ? matmul(dout, a, true, trans_a)
                   : matmul(a, dout, !trans_a, false);
}

} // namespace jittor
