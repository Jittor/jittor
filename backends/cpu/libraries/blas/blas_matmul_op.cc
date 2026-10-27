// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "blas_runtime.h"

#include "core/var.h"
#include "blas_matmul_op.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

#ifndef JIT

static auto make_blas_matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("blas_matmul");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");

BlasMatmulOp::BlasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    USER_CHECK(a->dtype() == b->dtype() && blas_gemm_dtype(a->dtype()))
        << "blas matmul takes two operands of one dtype (float32/float64/complex64/complex128),"
           " but got a:"
        << a->dtype() << "b:" << b->dtype();
    c = create_output(nullptr, a->dtype());
    set_flag(OpFlags::_cpu, 1);
    set_flag(OpFlags::_cuda, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
}

VarPtr BlasMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // c = op(a) @ op(b); the gradients come back in the operands' own
    // pre-transpose layouts. Complex: dA = dC op(B)^H and dB = op(A)^H dC,
    // torch's convention and the one cublas_matmul follows -- the other
    // operand enters conjugated, the transpose flags are those of a real GEMM.
    Var* other = v_index == 0 ? b : a;
    VarPtr other_conj;
    if (other->dtype().is_complex()) {
        other_conj = make_unary(other, ns_conj);
        other = other_conj.ptr;
    }
    if (v_index == 0) {
        if (trans_a)
            return make_blas_matmul(other, dout, trans_b, 1);
        return make_blas_matmul(dout, other, 0, trans_b^1);
    }
    if (trans_b)
        return make_blas_matmul(dout, other, 1, trans_a);
    return make_blas_matmul(other, dout, trans_a^1, 0);
}

void BlasMatmulOp::infer_shape() {
    USER_CHECKop(a->shape.size(),==,2);
    USER_CHECKop(b->shape.size(),==,2);
    // jit_run hands mem_ptr to CBLAS with the shape alone. The constructor
    // path makes operands dense, so this is the invariant, not a route.
    USER_CHECK(a->is_contiguous() && b->is_contiguous())
        << "blas matmul needs dense inputs; got strides" << a->storage_strides
        << "and" << b->storage_strides;
    int64 n = a->shape[0], m = a->shape[1];
    int64 m_ = b->shape[0], k = b->shape[1];
    if (trans_a) swap(n, m);
    if (trans_b) swap(m_, k);
    USER_CHECKop(m,==,m_) << "blas matmul inner dimensions must match";
    c->set_shape({n, k});
}

void BlasMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void BlasMatmulOp::jit_run() {
    const auto& as = a->shape;
    const auto& bs = b->shape;
    int64 n = as[0], m = as[1], k = bs[1];
    if ('@Trans_a'=='T') {
        n = as[1];
        m = as[0];
    }
    if ('@Trans_b'=='T')
        k = bs[0];
    blas_gemm_execute(a->dtype(), 1, n, m, k, '@Trans_a'=='T', '@Trans_b'=='T',
                      a->mem_ptr, b->mem_ptr, c->mem_ptr);
}
#endif
#endif // JIT

} // jittor
