// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_addmm_op.h"
#include "core/var.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

static auto make_mapped_matmul_addmm =
    op_constructor<VarPtr, Var*, Var*, bool, bool, bool>("mapped_matmul");
static auto make_reshape_addmm =
    op_constructor<VarPtr, Var*, NanoVector>("reshape");
static auto make_reduce_addmm =
    op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");

MappedAddmmOp::MappedAddmmOp(Var* x, Var* weight, Var* bias, bool allow_reduced_precision)
    : x(x), weight(weight), bias(bias), allow_reduced_precision(allow_reduced_precision) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    x->set_flag(VarFlags::_needed_by_backward);
    weight->set_flag(VarFlags::_needed_by_backward);
    USER_CHECK(x->dtype().is_float() && weight->dtype() == x->dtype() && bias->dtype() == x->dtype())
        << "mapped_addmm requires matching floating-point x, weight, bias; got"
        << x->dtype() << weight->dtype() << bias->dtype();
    y = create_output(nullptr, x->dtype());
}

void MappedAddmmOp::infer_shape() {
    USER_CHECK(x->shape.size() >= 2)
        << "mapped_addmm requires x with rank >= 2, got" << x->shape;
    USER_CHECK(weight->shape.size() == 2)
        << "mapped_addmm requires 2-D weight [N, K], got" << weight->shape;
    USER_CHECK(bias->shape.size() == 1)
        << "mapped_addmm requires 1-D bias [N], got" << bias->shape;
    const int64 k = x->shape[x->shape.size() - 1];
    const int64 n = weight->shape[0];
    USER_CHECK(weight->shape[1] == k)
        << "mapped_addmm K mismatch:" << k << "vs" << weight->shape[1];
    USER_CHECK(bias->shape[0] == n)
        << "mapped_addmm bias length mismatch:" << bias->shape[0] << "vs" << n;
    NanoVector out_shape;
    for (int i = 0; i + 1 < static_cast<int>(x->shape.size()); ++i)
        out_shape.push_back_check_overflow(x->shape[i]);
    out_shape.push_back_check_overflow(n);
    y->set_shape(out_shape);
}

VarPtr MappedAddmmOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    const int64 k = x->shape[x->shape.size() - 1];
    const int64 n = weight->shape[0];
    const int64 rows = x->num / (k ? k : 1);
    if (v_index == 0) {
        if (x->shape.size() == 2)
            return make_mapped_matmul_addmm(dout, weight, 0, 0, allow_reduced_precision);
        auto dout2 = make_reshape_addmm(dout, NanoVector{rows, n});
        auto dx2 = make_mapped_matmul_addmm(dout2.ptr, weight, 0, 0, allow_reduced_precision);
        return make_reshape_addmm(dx2.ptr, x->shape);
    }
    if (v_index == 1) {
        if (x->shape.size() == 2)
            return make_mapped_matmul_addmm(dout, x, 1, 0, allow_reduced_precision);
        auto dout2 = make_reshape_addmm(dout, NanoVector{rows, n});
        auto x2 = make_reshape_addmm(x, NanoVector{rows, k});
        return make_mapped_matmul_addmm(dout2.ptr, x2.ptr, 1, 0, allow_reduced_precision);
    }
    if (v_index == 2) {
        NanoVector axes;
        for (int i = 0; i + 1 < static_cast<int>(dout->shape.size()); ++i)
            axes.push_back(i);
        return make_reduce_addmm(dout, ns_add, axes, false);
    }
    return nullptr;
}

void MappedAddmmOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype() << "«R:" << x->shape.size();
}

#else
void MappedAddmmOp::jit_run() {
    USER_ERROR << "mapped_addmm is only available through a mapped backend";
}
#endif

} // jittor
