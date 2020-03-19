// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "cublas_matmul_op.h"
#include "cublas_warper.h"

using namespace std;

namespace jittor {

#ifndef JIT

CublasMatmulOp::CublasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == b->dtype().dsize()) << "type of two inputs should be the same";
    c = create_output(nullptr, a->dtype());
}

void CublasMatmulOp::infer_shape() {
    ASSERTop(a->shape.size(),==,2);
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

void CublasMatmulOp::jit_prepare() {
    add_jit_define("T", a->dtype());
    add_jit_define("Trans_a", trans_a ? "T" : "N");
    add_jit_define("Trans_b", trans_b ? "T" : "N");
    add_jit_define("op", a->dtype().dsize() == 4 ? "S" : "D");
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void CublasMatmulOp::jit_run() {
    cublasHandle_t& handle_ = cublas_handle;
    const T alpha = 1.0f;
    const T beta  = 0.0f;

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
    // a: [n,m], b: [m,k], c: [n,k]
    checkCudaErrors(cublas@op@@gemm(handle_, 
    CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a, 
    k, n, m, &alpha, 
    b->ptr<T>(), '@Trans_b' == 'N' ? k : m, 
    a->ptr<T>(), '@Trans_a' == 'N' ? m : n, &beta, 
    c->ptr<T>(), k));
}
#endif
#endif // JIT

} // jittor
