// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "core/var.h"
#include "cublas_matmul_op.h"
#include "cublas_wrapper.h"
#include "cublas_compute_type.h"

using namespace std;

namespace jittor {

#ifndef JIT

static auto make_cublas_matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("cublas_matmul");

CublasMatmulOp::CublasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    // TODO: support int8 * int8
    USER_CHECK(a->dtype().is_float() && b->dtype().is_float())
        << "cublas matmul requires floating-point inputs (float16/float32/float64), but got a:"
        << a->dtype() << "b:" << b->dtype();
    // TODO: support diffrent input type
    USER_CHECK(a->dtype().dsize() == b->dtype().dsize())
        << "matmul inputs must have the same dtype, but got a:" << a->dtype() << "b:" << b->dtype();
    c = create_output(nullptr, a->dtype());
}

VarPtr CublasMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // c = op(a) @ op(b). Return gradients in the original, pre-transpose
    // layouts so explicit cuBLAS fast paths remain differentiable.
    if (v_index == 0) {
        if (trans_a)
            return make_cublas_matmul(b, dout, trans_b, 1);
        return make_cublas_matmul(dout, b, 0, trans_b^1);
    }
    if (trans_b)
        return make_cublas_matmul(dout, a, 1, trans_a);
    return make_cublas_matmul(a, dout, trans_a^1, 0);
}

// An operand of rank > 2 is the same buffer as its rank-2 flattening: a dense
// row-major `(d0, .., dn-1, m)` is `(d0*..*dn-1, m)`, same pointer, same leading
// dimension. cuBLAS only ever sees the flattened extents, so accepting the
// higher rank here costs nothing at the kernel and saves the caller two pure
// view nodes per call -- `matmul` used to reshape into rank 2 and back out,
// which was 64 of the 408 graph nodes in a transformer decode step.
static void flatten_2d(const NanoVector& shape, int64& rows, int64& cols) {
    rows = 1;
    for (uint i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
    cols = shape[shape.size() - 1];
}

void CublasMatmulOp::infer_shape() {
    USER_CHECKop(a->shape.size(),>=,2)
        << "cublas matmul requires rank-2 or higher input a, got rank " << a->shape.size();
    USER_CHECKop(b->shape.size(),>=,2)
        << "cublas matmul requires rank-2 or higher input b, got rank " << b->shape.size();
    int64 an, am, bn, bm;
    flatten_2d(a->shape, an, am);
    flatten_2d(b->shape, bn, bm);
    // after op(): n x m times m_ x k
    int64 n = an, m = am, m_ = bn, k = bm;
    if (trans_a) swap(n, m);
    if (trans_b) swap(m_, k);
    USER_CHECKop(m,==,m_)
        << "cublas matmul inner dimensions must match, but got " << m << " and " << m_;
    // Without a transpose the result's rows are a's leading axes, so they keep
    // their individual extents; with one they are a's last axis and the result
    // is rank 2. Either way the row count is `n`, which is what the kernel and
    // the allocation see.
    if (!trans_a && a->shape.size() > 2) {
        NanoVector cshape;
        for (uint i = 0; i + 1 < a->shape.size(); ++i) cshape.push_back(a->shape[i]);
        cshape.push_back(k);
        c->set_shape(cshape);
    } else {
        c->set_shape({n, k});
    }
}

void CublasMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
    jk << "«op:" << (a->dtype().dsize() == 2? 'H' : (a->dtype().dsize() == 4 ? 'S' : 'D'));
}

#else // JIT
#pragma clang diagnostic ignored "-Wtautological-compare"

void CublasMatmulOp::jit_run() {
    cublasHandle_t handle_ = cublas_bind_stream();
    const T alpha = 1.0f;
    const T beta  = 0.0f;
    const float alpha_f = 1.0f;
    const float beta_f  = 0.0f;
    void* alpha_p = (void*)&alpha_f;
    void* beta_p = (void*)&beta_f;

    // The flattened extents, not shape[0]/shape[1]: an operand of rank > 2 is
    // the same dense buffer as its rank-2 flattening (see infer_shape).
    const auto& as = a->shape;
    const auto& bs = b->shape;
    int64 an = 1, bn = 1;
    for (uint i = 0; i + 1 < as.size(); ++i) an *= as[i];
    for (uint i = 0; i + 1 < bs.size(); ++i) bn *= bs[i];
    auto am = as[as.size()-1];
    auto bm = bs[bs.size()-1];
    auto n = an;
    auto m = am;
    auto k = bm;
    if ('@Trans_a'=='T') {
        n = am;
        m = an;
    }
    if ('@Trans_b'=='T') {
        k = bn;
    }
    bool has_fp16 = a->dtype() == ns_float16
        || b->dtype() == ns_float16 || c->dtype() == ns_float16;
    bool has_bf16 = a->dtype() == ns_bfloat16
        || b->dtype() == ns_bfloat16 || c->dtype() == ns_bfloat16;
    bool has_fp64 = a->dtype() == ns_float64
        || b->dtype() == ns_float64 || c->dtype() == ns_float64;
    // a: [n,m], b: [m,k], c: [n,k]
    CublasGemmMode mode = cublas_gemm_mode(has_fp16, has_bf16, has_fp64);
    auto computeType = mode.compute;
    auto algo = mode.algo;
    if (mode.typed_alpha) {
        alpha_p = (void*)&alpha;
        beta_p = (void*)&beta;
    }
    LOGvvv << "cublas_matmul algo select:"
        << "precision=" >> float32_precision_tier_name(mode.tier)
        << "computeType=" >> cublas_compute_type_name(computeType)
        << "algo=" >> cublas_gemm_algo_name(algo);
    checkCudaErrors(cublasGemmEx(handle_, 
    CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a, 
    k, n, m, alpha_p, 
    b->ptr<T>(),get_dtype(b->dtype()), '@Trans_b' == 'N' ? k : m, 
    a->ptr<T>(),get_dtype(a->dtype()), '@Trans_a' == 'N' ? m : n, beta_p, 
    c->ptr<T>(),get_dtype(c->dtype()), k,
    computeType, algo));
    // checkCudaErrors(cublas@op@@gemm(handle_, 
    // CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a, 
    // k, n, m, &alpha, 
    // b->ptr<T>(), '@Trans_b' == 'N' ? k : m, 
    // a->ptr<T>(), '@Trans_a' == 'N' ? m : n, &beta, 
    // c->ptr<T>(), k));

    
}
#endif // JIT

} // jittor
