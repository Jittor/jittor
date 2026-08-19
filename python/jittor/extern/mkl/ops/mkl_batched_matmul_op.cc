// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <dnnl.hpp>

#include "var.h"
#include "mkl_batched_matmul_op.h"

using namespace dnnl;
using namespace std;

namespace jittor {

#ifndef JIT

static auto make_mkl_batched_matmul = get_op_info("mkl_batched_matmul")
    .get_constructor<VarPtr, Var*, Var*, bool, bool>();

MklBatchedMatmulOp::MklBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    ASSERT(a->dtype().is_float() && b->dtype().is_float())
        << "mkl batched matmul requires floating-point inputs, but got a:"
        << a->dtype() << "b:" << b->dtype();
    ASSERT(a->dtype().dsize() == 4 && b->dtype().dsize() == 4)
        << "mkl batched matmul supports float32 only, but got a:"
        << a->dtype() << "b:" << b->dtype();
    c = create_output(nullptr, a->dtype());
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    a->flags.set(NodeFlags::_needed_by_backward);
    b->flags.set(NodeFlags::_needed_by_backward);
}

VarPtr MklBatchedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // a [..,n,m], b [..,m,k], c [..,n,k], c = a*b
    if (v_index == 0) {
        if (trans_a)
            return make_mkl_batched_matmul(b, dout, trans_b, 1);
        else
            // da = dc*b^T
            return make_mkl_batched_matmul(dout, b, 0, trans_b^1);
    } else {
        if (trans_b)
            return make_mkl_batched_matmul(dout, a, 1, trans_a);
        else
            // db = a^T*dc
            return make_mkl_batched_matmul(a, dout, trans_a^1, 0);
    }
}

void MklBatchedMatmulOp::infer_shape() {
    auto adim = a->shape.size();
    auto bdim = b->shape.size();
    ASSERTop(adim,>=,3);
    ASSERTop(bdim,>=,3);
    ASSERTop(adim,==,bdim);

    auto n = a->shape[adim-2], m = a->shape[adim-1];
    auto m_ = b->shape[bdim-2], k = b->shape[bdim-1];

    NanoVector c_shape;
    for (uint i=0; i<adim-2; i++) {
        ASSERTop(a->shape[i],==,b->shape[i]);
        c_shape.push_back(a->shape[i]);
    }
    if (trans_a) swap(n, m);
    if (trans_b) swap(m_, k);
    ASSERTop(m,==,m_);
    c_shape.push_back(n);
    c_shape.push_back(k);

    c->set_shape(c_shape);
}

void MklBatchedMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void MklBatchedMatmulOp::jit_run() {
    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto adim = as.size();
    memory::dim batch_size = 1;
    for (uint i=0; i+2<adim; i++)
        batch_size *= as[i];
    memory::dim n = as[adim-2];
    memory::dim m = as[adim-1];
    memory::dim k = bs[adim-1];
    if ('@Trans_a'=='T') {
        n = as[adim-1];
        m = as[adim-2];
    }
    if ('@Trans_b'=='T') {
        k = bs[adim-2];
    }

    // One gemm per matrix, parallel across the batch. Attention matrices are
    // only a few hundred rows, so a single batched primitive leaves oneDNN
    // splitting one small problem over every core; giving each core a whole
    // matrix scales far better. Nested threading is off by default, so each
    // dnnl_sgemm here runs sequentially.
    auto lda = ('@Trans_a'=='N') ? m : n;
    auto ldb = ('@Trans_b'=='N') ? k : m;
    auto* ap = a->ptr<T>();
    auto* bp = b->ptr<T>();
    auto* cp = c->ptr<T>();
    #pragma omp parallel for schedule(static)
    for (int64 i=0; i<batch_size; i++) {
        dnnl_sgemm('@Trans_a', '@Trans_b', n, k, m,
            1.f, ap + i*n*m, lda,
            bp + i*m*k, ldb,
            0.f, cp + i*n*k, k);
    }
}
#endif
#endif // JIT

} // jittor
