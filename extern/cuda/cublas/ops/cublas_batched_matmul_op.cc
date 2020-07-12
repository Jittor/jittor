// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Meng-Hao Guo <guomenghao1997@gmail.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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
        // da = dc*b^T
        return make_cublas_batched_matmul(dout, b, trans_a^0, trans_b^1);
    } else {
        // db = a^T*dc
        return make_cublas_batched_matmul(a, dout, trans_a^1, trans_b^0);
    }
}

void CublasBatchedMatmulOp::infer_shape(){
    // TODO: 改成bmm的infer shape
    ASSERTop(a->shape.size(),==,3);
    ASSERTop(b->shape.size(),==,3);

    int batch_size = a->shape[0], n = a->shape[1], m = a->shape[2];
    int batch_size_ = b->shape[0], m_ = b->shape[1], k = b->shape[2];

    ASSERTop(batch_size,==,batch_size_);
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    ASSERTop(m,==,m_);

    c->set_shape({batch_size, n, k});
}

void CublasBatchedMatmulOp::jit_prepare() {
    add_jit_define("T", a->dtype());
    add_jit_define("Trans_a", trans_a ? "T" : "N");
    add_jit_define("Trans_b", trans_b ? "T" : "N");
    add_jit_define("op", a->dtype().dsize() == 4 ? "S" : "D");
}

#else // JIT
#ifdef JIT_cuda
#pragma clang diagnostic ignored "-Wtautological-compare"
void CublasBatchedMatmulOp::jit_run() {
    // TODO
    cublasHandle_t& handle_ = cublas_handle;
    const T alpha = 1.0f;
    const T beta  = 0.0f;

    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto batch_size = as[0];
    auto n = as[1];
    auto m = as[2];
    auto k = bs[2];
    if ('@Trans_a'=='T') {
        n = as[2];
        m = as[1];
    }
    if ('@Trans_b'=='T') {
        k = bs[1];
    }
    // a: [b,n,m], b: [b,m,k], c: [b,n,k]
    // 修改成bmm接口
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


