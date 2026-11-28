// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_unary_op.h"
#include "core/var.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");
static auto make_mapped_unary_self = op_constructor<VarPtr, Var*, int, double, double>("mapped_unary");
static auto make_mapped_unary_grad = op_constructor<VarPtr, Var*, Var*, int, double>("mapped_unary_grad");
static auto make_reduce_sum_axis = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");

MappedUnaryOp::MappedUnaryOp(Var* x, int kind, double param0, double param1)
    : x(x), kind(kind), param0(param0), param1(param1) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    USER_CHECK(kind >= kRsqrt && kind <= kRelu)
        << "mapped_unary: unsupported kind" << kind;
    if (kind == kRsqrt || kind == kSilu || kind == kSoftplus || kind == kSoftmax || kind == kGelu) {
        USER_CHECK(x->dtype().is_float())
            << "mapped_unary requires floating-point input, got" << x->dtype();
    }
    if (kind == kSoftplus) {
        USER_CHECK(param0 > 0.0) << "mapped_unary softplus requires beta > 0";
    }
    if (kind == kSilu || kind == kSoftplus || kind == kRelu || kind == kGelu)
        x->set_flag(VarFlags::_needed_by_backward);
    y = create_output(nullptr, x->dtype());
    if (kind == kRsqrt || kind == kSoftmax)
        y->set_flag(VarFlags::_needed_by_backward);
}

void MappedUnaryOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr MappedUnaryOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index != 0) return nullptr;
    if (kind == kRsqrt) {
        // d(x^(-1/2))/dx = -0.5 * out^3
        auto half_neg = make_number(-0.5f, out);
        auto out2 = make_binary(out, out, ns_multiply);
        auto out3 = make_binary(out2.ptr, out, ns_multiply);
        auto scale = make_binary(half_neg.ptr, out3.ptr, ns_multiply);
        return make_binary(dout, scale.ptr, ns_multiply);
    }
    if (kind == kSilu) {
        // d(x * sig(x))/dx = sig(x) * (1 + x * (1 - sig(x)))
        auto sig = make_unary(x, ns_sigmoid);
        auto one = make_number(1.0f, x);
        auto one_minus_sig = make_binary(one.ptr, sig.ptr, ns_subtract);
        auto term = make_binary(x, one_minus_sig.ptr, ns_multiply);
        auto inner = make_binary(one.ptr, term.ptr, ns_add);
        auto dsilu = make_binary(sig.ptr, inner.ptr, ns_multiply);
        return make_binary(dout, dsilu.ptr, ns_multiply);
    }
    if (kind == kTriu || kind == kTril) {
        return make_mapped_unary_self(dout, kind, param0, 0.0);
    }
    if (kind == kSoftmax) {
        int dim = static_cast<int>(param0);
        if (dim < 0) dim += static_cast<int>(out->shape.size());
        auto prod = make_binary(dout, out, ns_multiply);
        auto summed = make_reduce_sum_axis(prod.ptr, ns_add, NanoVector{dim}, true);
        auto diff = make_binary(dout, summed.ptr, ns_subtract);
        return make_binary(out, diff.ptr, ns_multiply);
    }
    if (kind == kCumsum) {
        USER_ERROR << "mapped_unary cumsum backward is not supported; use CumsumACL in grad mode";
        return nullptr;
    }
    if (kind == kGelu) {
        return make_mapped_unary_grad(dout, x, kGelu, param0);
    }
    if (kind == kRelu) {
        return make_mapped_unary_grad(dout, x, kRelu, 0.0);
    }
    // kSoftplus: d/dx = 1 if beta*x > threshold else sigmoid(beta*x)
    VarPtr bx = x;
    if (param0 != 1.0) {
        auto beta = make_number(static_cast<float>(param0), x);
        bx = make_binary(x, beta.ptr, ns_multiply);
    }
    auto thr = make_number(static_cast<float>(param1), x);
    auto above = make_binary(bx.ptr, thr.ptr, ns_greater);
    auto sig = make_unary(bx.ptr, ns_sigmoid);
    auto dsoft = make_binary(dout, sig.ptr, ns_multiply);
    return make_ternary(above.ptr, dout, dsoft.ptr);
}

void MappedUnaryOp::jit_prepare(JK& jk) {
    jk << "«K:" << kind << "«T:" << x->dtype();
}

#else
void MappedUnaryOp::jit_run() {
    USER_ERROR << "mapped_unary is only available through a mapped backend";
}
#endif

} // jittor
