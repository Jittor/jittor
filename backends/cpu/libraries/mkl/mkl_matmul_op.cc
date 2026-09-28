// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "onednn_runtime.h"

#include "core/var.h"
#include "mkl_matmul_op.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

#ifndef JIT

static auto make_mkl_matmul = op_constructor<VarPtr, Var*, Var*, bool, bool>("mkl_matmul");

MklMatmulOp::MklMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    // TODO: support int8 * int8
    USER_CHECK(a->dtype().is_float() && b->dtype().is_float())
        << "mkl matmul requires floating-point inputs, but got a:" << a->dtype() << "b:" << b->dtype();
    // TODO: support diffrent input type
    USER_CHECK(a->dtype().dsize() == 4 && b->dtype().dsize() == 4) << "support float32 only now.";
    c = create_output(nullptr, a->dtype());
    // Both flags used to be someone else's business: this op only ever entered
    // a graph as a relay inside a fused op, where autograd runs on the
    // meta-op subgraph the relay stands in for, so it needed neither a
    // gradient nor its operands kept alive for one. It is a forward-graph op
    // now (the CPU row of the matmul kernel table), and a forward-graph op
    // without these silently produces a wrong gradient rather than an error.
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
}

VarPtr MklMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // a [n,m], b [m,k], c [n,k], c = op(a) * op(b). The gradients come back in
    // the operands' own pre-transpose layouts. Mirrors MklBatchedMatmulOp.
    if (v_index == 0) {
        if (trans_a)
            return make_mkl_matmul(b, dout, trans_b, 1);
        // da = dc * b^T
        return make_mkl_matmul(dout, b, 0, trans_b^1);
    }
    if (trans_b)
        return make_mkl_matmul(dout, a, 1, trans_a);
    // db = a^T * dc
    return make_mkl_matmul(a, dout, trans_a^1, 0);
}

void MklMatmulOp::infer_shape() {
    USER_CHECKop(a->shape.size(),==,2);
    USER_CHECKop(b->shape.size(),==,2);
    // jit_run hands mem_ptr to oneDNN with the shape alone, so a strided
    // operand would be read as though it were dense. The kernel table's
    // predicate keeps views off the forward path, but grad() builds operands
    // of its own out of a cotangent, and a cotangent can arrive as a view --
    // so the contract is enforced here, where every route passes.
    USER_CHECK(a->is_contiguous())
        << "mkl matmul needs a dense input a; got strides"
        << a->storage_strides << "for shape" << a->shape
        << "(call contiguous() first)";
    USER_CHECK(b->is_contiguous())
        << "mkl matmul needs a dense input b; got strides"
        << b->storage_strides << "for shape" << b->shape
        << "(call contiguous() first)";
    int n = a->shape[0], m = a->shape[1];
    int m_ = b->shape[0], k = b->shape[1];
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    USER_CHECKop(m,==,m_);
    c->set_shape({n, k});
}

void MklMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void MklMatmulOp::jit_run() {
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
    onednn_matmul_execute(1, n, m, k, '@Trans_a'=='T', '@Trans_b'=='T',
                           a->mem_ptr, b->mem_ptr, c->mem_ptr);
}
#endif
#endif // JIT

} // jittor
