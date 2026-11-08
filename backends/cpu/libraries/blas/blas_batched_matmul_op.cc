// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "blas_runtime.h"

#include "core/var.h"
#include "blas_batched_matmul_op.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

#ifndef JIT

static auto make_blas_batched_matmul =
    op_constructor<VarPtr, Var*, Var*, bool, bool>("blas_batched_matmul");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");

BlasBatchedMatmulOp::BlasBatchedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    USER_CHECK(a->dtype() == b->dtype() && blas_gemm_dtype(a->dtype()))
        << "blas batched matmul takes two operands of one dtype (float32/float64/complex64/complex128),"
           " but got a:" << a->dtype() << "b:" << b->dtype();
    c = create_output(nullptr, a->dtype());
    set_flag(OpFlags::_cpu, 1);
    set_flag(OpFlags::_cuda, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
}

VarPtr BlasBatchedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // As BlasMatmulOp::grad, per batch: the other operand enters conjugated.
    Var* other = v_index == 0 ? b : a;
    VarPtr other_conj;
    if (other->dtype().is_complex()) {
        other_conj = make_unary(other, ns_conj);
        other = other_conj.ptr;
    }
    if (v_index == 0) {
        if (trans_a)
            return make_blas_batched_matmul(other, dout, trans_b, 1);
        return make_blas_batched_matmul(dout, other, 0, trans_b^1);
    }
    if (trans_b)
        return make_blas_batched_matmul(dout, other, 1, trans_a);
    return make_blas_batched_matmul(other, dout, trans_a^1, 0);
}

void BlasBatchedMatmulOp::infer_shape() {
    auto adim = a->shape.size();
    auto bdim = b->shape.size();
    USER_CHECKop(adim,>=,3);
    USER_CHECKop(adim,==,bdim);
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "blas batched matmul needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
    auto n = a->shape[adim-2], m = a->shape[adim-1];
    auto m_ = b->shape[bdim-2], k = b->shape[bdim-1];
    NanoVector c_shape;
    for (uint i=0; i<adim-2; i++) {
        // One batch stride per operand: broadcast batch dims are expanded by
        // the caller (`_broadcast_batch_dims`) before they reach here.
        USER_CHECKop(a->shape[i],==,b->shape[i]);
        c_shape.push_back_check_overflow(a->shape[i]);
    }
    if (trans_a) swap(n, m);
    if (trans_b) swap(m_, k);
    USER_CHECKop(m,==,m_) << "blas batched matmul inner dimensions must match";
    c_shape.push_back_check_overflow(n);
    c_shape.push_back_check_overflow(k);
    c->set_shape(c_shape);
}

void BlasBatchedMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void BlasBatchedMatmulOp::jit_run() {
    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto adim = as.size();
    int64 batch = 1;
    for (uint i=0; i+2<adim; i++)
        batch *= as[i];
    int64 n = as[adim-2], m = as[adim-1], k = bs[adim-1];
    if ('@Trans_a'=='T') {
        n = as[adim-1];
        m = as[adim-2];
    }
    if ('@Trans_b'=='T')
        k = bs[adim-2];
    blas_gemm_execute(a->dtype(), batch, n, m, k, '@Trans_a'=='T', '@Trans_b'=='T',
                      a->mem_ptr, b->mem_ptr, c->mem_ptr);
}
#endif
#endif // JIT

} // jittor
