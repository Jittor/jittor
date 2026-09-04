// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
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

void CublasMatmulOp::infer_shape() {
    USER_CHECKop(a->shape.size(),==,2);
    ASSERTop(b->shape.size(),==,2);
    int n = a->shape[0], m = a->shape[1];
    int m_ = b->shape[0], k = b->shape[1];
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    ASSERTop(m,==,m_);
    c->set_shape({n, k});
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

    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto n = as[0];
    auto m = as[1];
    auto k = bs[1];
    if ('@Trans_a'=='T') {
        n = as[1];
        m = as[0];
    }
    if ('@Trans_b'=='T') {
        k = bs[0];
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
