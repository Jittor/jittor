// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Meng-Hao Guo <guomenghao1997@gmail.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************


// cublas_batched_matmul_op.cc
#include "var.h"

#include "cublas_batched_matmul_op.h"
#include "cublas_warper.h"

using namespace std;

namespace jittor {

#ifndef JIT

static auto make_cublas_batched_matmul = get_op_info("cublas_batched_matmul")
    .get_constructor<VarPtr, Var*, Var*, bool, bool>();

CublasBatchedMatmulOp::CublasBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == b->dtype().dsize()) << "type of two inputs should be the same";
    c = create_output(nullptr, a->dtype());
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
}


VarPtr CublasBatchedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // a [b,n,m] b [b,m,k], c[b,n,k]
    // c = a*b
    if (v_index == 0) {
        if (trans_a)
            return make_cublas_batched_matmul(b, dout, trans_b, 1);
        else
            // da = dc*b^T
            return make_cublas_batched_matmul(dout, b, 0, trans_b^1);
    } else {
        if (trans_b)
            return make_cublas_batched_matmul(dout, a, 1, trans_a);
        else
            // db = a^T*dc
            return make_cublas_batched_matmul(a, dout, trans_a^1, 0);
    }
}

void CublasBatchedMatmulOp::infer_shape(){
    auto adim = a->shape.size();
    auto bdim = b->shape.size();
    ASSERTop(adim,>=,3);
    ASSERTop(bdim,>=,3);
    ASSERTop(adim,==,bdim);

    auto n = a->shape[adim-2], m = a->shape[adim-1];
    auto m_ = b->shape[adim-2], k = b->shape[adim-1];

    NanoVector c_shape;

    for (int i=0; i<adim-2; i++) {
        ASSERTop(a->shape[i],==,b->shape[i]);
        c_shape.push_back(a->shape[i]);
    }
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    ASSERTop(m,==,m_);
    c_shape.push_back(n);
    c_shape.push_back(k);

    c->set_shape(c_shape);
}

void CublasBatchedMatmulOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << a->dtype();
    jk << _CS("][Trans_a:") << (trans_a ? 'T' : 'N');
    jk << _CS("][Trans_b:") << (trans_b ? 'T' : 'N');
    jk << _CS("][op:") << (a->dtype().dsize() == 4 ? 'S' : 'D');
    jk << ']';
}

#else // JIT
#ifdef JIT_cuda
#pragma clang diagnostic ignored "-Wtautological-compare"
void CublasBatchedMatmulOp::jit_run() {
    cublasHandle_t& handle_ = cublas_handle;
    const T alpha = 1.0f;
    const T beta  = 0.0f;

    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto adim = as.size();
    auto batch_size = as[0];
    for (int i=1; i<adim-2; i++)
        batch_size *= as[i];
    auto n = as[adim-2];
    auto m = as[adim-1];
    auto k = bs[adim-1];
    if ('@Trans_a'=='T') {
        n = as[adim-1];
        m = as[adim-2];
    }
    if ('@Trans_b'=='T') {
        k = bs[adim-2];
    }
    // a: [b,n,m], b: [b,m,k], c: [b,n,k]
    checkCudaErrors(cublas@op@@gemmStridedBatched(handle_,
    CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a,
    k, n, m, &alpha,
    b->ptr<T>(), '@Trans_b' == 'N' ? k : m, k * m, 
    a->ptr<T>(), '@Trans_a' == 'N' ? m : n, n * m, &beta,
    c->ptr<T>(), k, k * n,
    batch_size));
}
#endif
#endif // JIT

} // jittor


