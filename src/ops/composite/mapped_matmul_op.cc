// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_matmul_op.h"
#include "core/var.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

static auto make_mapped_matmul =
    op_constructor<VarPtr, Var*, Var*, bool, bool, bool>("mapped_matmul");

MappedMatmulOp::MappedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b, bool allow_reduced_precision) : a(a), b(b), trans_a(trans_a), trans_b(trans_b), allow_reduced_precision(allow_reduced_precision) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    a->set_flag(VarFlags::_needed_by_backward);
    b->set_flag(VarFlags::_needed_by_backward);
    USER_CHECK(a->dtype().is_float() && b->dtype().is_float())
        << "mapped_matmul requires floating-point operands, but got a:"
        << a->dtype() << "b:" << b->dtype();
    USER_CHECK(a->dtype() == b->dtype())
        << "mapped_matmul operands must have the same dtype, but got a:"
        << a->dtype() << "b:" << b->dtype();
    // Both transposed is unreachable, not merely unsupported: a forward is
    // built with at most one transpose and the gradient below preserves that.
    // Saying so here keeps a backend descriptor -- which can present exactly
    // one operand transposed -- from having to invent an answer.
    USER_CHECK(!(trans_a && trans_b))
        << "mapped_matmul transposes at most one operand; transpose the other"
        << "one in the graph instead";
    c = create_output(nullptr, a->dtype());
}

VarPtr MappedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // c = A @ B with A = op(a) and B = op(b); dA = dout @ B^T and
    // dB = A^T @ dout, and a transposed operand receives the transpose of its
    // own gradient. Written back in the pre-transpose layout, so no transpose
    // node is introduced, and with the reduced-precision setting of the
    // forward rather than of whenever the backward is built.
    if (v_index == 0) {
        if (trans_a)
            return make_mapped_matmul(b, dout, trans_b, 1, allow_reduced_precision);
        return make_mapped_matmul(dout, b, 0, trans_b^1, allow_reduced_precision);
    }
    if (trans_b)
        return make_mapped_matmul(dout, a, 1, trans_a, allow_reduced_precision);
    return make_mapped_matmul(a, dout, trans_a^1, 0, allow_reduced_precision);
}

void MappedMatmulOp::infer_shape() {
    auto adim = a->shape.size();
    auto bdim = b->shape.size();
    USER_CHECKop(adim,>=,2) << "mapped_matmul operands are matrices; a is" << a->shape;
    // Equal ranks, deliberately. The frontend already folds every other case:
    // a 1-D operand is reshaped into a matrix, and a stack of matrices times
    // one matrix is flattened into a single 2-D product. Accepting mixed ranks
    // here would mean carrying a broadcast rule no caller reaches and that the
    // gradient would then have to undo.
    USER_CHECK(adim == bdim)
        << "mapped_matmul operands must have equal rank, but got a:" << a->shape
        << "b:" << b->shape;
    int64 n = a->shape[adim-2], m = a->shape[adim-1];
    int64 m_ = b->shape[bdim-2], k = b->shape[bdim-1];
    if (trans_a) std::swap(n, m);
    if (trans_b) std::swap(m_, k);
    USER_CHECK(m == m_)
        << "mapped_matmul contracted dims must be equal, but got" << m << "and" << m_
        << "for a:" << a->shape << "b:" << b->shape
        << "with trans_a:" << trans_a << "trans_b:" << trans_b;
    NanoVector shape;
    for (uint i = 0; i + 2 < adim; i++) {
        USER_CHECK(a->shape[i] == b->shape[i])
            << "mapped_matmul batch dims must be equal, but got a:" << a->shape
            << "b:" << b->shape << "-- broadcast them before the product";
        shape.push_back(a->shape[i]);
    }
    shape.push_back(n);
    shape.push_back(k);
    c->set_shape(shape);
}

void MappedMatmulOp::jit_prepare(JK& jk) {
    // The op has one native implementation per backend and no generated body,
    // so this fragment exists to keep it a JIT-keyed op rather than to select
    // a kernel: an op with an empty fragment is executed through the native
    // callback directly, and `Profiler::record_and_run` -- everything
    // `jt.profiler` and bench/launch_count.py report -- only sees the keyed
    // path. Measured back to back against the keyless variant on Ascend950PR,
    // the difference was inside the noise of a shared device (12.8 vs 13.0
    // us/launch for a 32x32 product), so the launch census is free.
    jk << "«T:" << a->dtype() << "«R:" << a->shape.size()
       << "«Ta:" << (trans_a ? 'T' : 'N') << "«Tb:" << (trans_b ? 'T' : 'N');
}

#else
void MappedMatmulOp::jit_run() {
    USER_ERROR << "mapped_matmul is only available through a mapped backend";
}
#endif

} // jittor
