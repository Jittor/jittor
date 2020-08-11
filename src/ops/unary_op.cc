// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/unary_op.h"
#include "ops/unary_op_defs.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_unary = get_op_info("unary")
    .get_constructor<VarPtr, Var*, NanoString>();
static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();

static unordered_set<string> unary_ops = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    // please keep float64 the last type
    // @pybind(abs, __abs__)
    "abs",
    // @pybind(negative, __neg__)
    "negative",
    "logical_not",
    "bitwise_not",
    "log",
    "exp",
    "sqrt",
    "round",
    "floor",
    "ceil",
    "sin",
    // @pybind(asin, arcsin)
    "asin",
    "sinh",
    // @pybind(asinh, arcsinh)
    "asinh",
    "tan",
    // @pybind(atan, arctan)
    "atan",
    "tanh",
    // @pybind(atanh, arctanh)
    "atanh",
    "cos",
    // @pybind(acos, arccos)
    "acos",
    "cosh",
    // @pybind(acosh, arccosh)
    "acosh",
    "sigmoid",
};

UnaryOp::UnaryOp(Var* x, NanoString op) : x(x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    ns = op;
    ASSERT(ns.is_unary() | ns.is_dtype());
    NanoString dtype;
    if (ns.is_dtype()) {
        if (ns == x->dtype()) {
            forward(x);
            return;
        }
        dtype = ns;
        ns = ns_cast;
    } else if (ns.is_bool())
        dtype = ns_bool;
    else if (ns.is_float())
        dtype = dtype_infer(x->ns, x->ns, 2);
    else if (ns.is_int())
        dtype = dtype_infer(x->ns, x->ns, 1);
    else {
        dtype = x->ns;
    }
    y = create_output(nullptr, dtype);
}

VarPtr UnaryOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (!x->is_float()) return nullptr;
    if (ns == ns_cast) return make_unary(dout, x->dtype());
    if (ns == ns_negative) return make_unary(dout, ns);
    if (ns == ns_abs) {
        auto neg = make_unary(dout, ns_negative);
        auto zeros = make_number(0, x);
        auto cond = make_binary(x, zeros, ns_greater_equal);
        return make_ternary(cond, dout, neg);
    }
    if (ns == ns_log)
        return make_binary(dout, x, ns_divide);
    if (ns == ns_exp)
        return make_binary(dout, y, ns_multiply);
    if (ns == ns_sqrt){
        auto two = make_number(2, x);
        auto twoy = make_binary(two, y, ns_multiply);
        return make_binary(dout, twoy, ns_divide);
    }
    // dsin(x) = cos(x)
    if (ns == ns_sin)
        return make_binary(dout, make_unary(x, ns_cos), ns_multiply);
    // dcos(x) = -sin(x)
    if (ns == ns_cos)
        return make_binary(dout, make_unary(make_unary(x, ns_sin), ns_negative), ns_multiply);
    // dtan(x) = 1/cos^2(x)
    if (ns == ns_tan) {
        auto one = make_number(1, x);
        auto cosx = make_unary(x, ns_cos);
        auto cos2x = make_binary(cosx, cosx, ns_multiply);
        return make_binary(dout, cos2x, ns_divide);
    }
    // dasin(x) = 1/sqrt(1-x^2)
    if (ns == ns_asin) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(one, x2, ns_subtract);
        x2 = make_unary(x2, ns_sqrt);
        return make_binary(dout, x2, ns_divide);
    }
    // dacos(x) = -1/sqrt(1-x^2)
    if (ns == ns_acos) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(one, x2, ns_subtract);
        x2 = make_unary(x2, ns_sqrt);
        return make_unary(make_binary(dout, x2, ns_divide), ns_negative);
    }
    // datan(x) = 1/(x^2+1)
    if (ns == ns_atan) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(one, x2, ns_add);
        return make_binary(dout, x2, ns_divide);
    }

    // dsinh(x) = cosh(x)
    if (ns == ns_sinh)
        return make_binary(dout, make_unary(x, ns_cosh), ns_multiply);
    // dcosh(x) = sinh(x)
    if (ns == ns_cosh)
        return make_binary(dout, make_unary(x, ns_sinh), ns_multiply);
    // dtanh(x) = 1/cosh^2(x)
    if (ns == ns_tanh) {
        auto cosx = make_unary(x, ns_cosh);
        auto cos2x = make_binary(cosx, cosx, ns_multiply);
        return make_binary(dout, cos2x, ns_divide);
    }

    // dasinh(x) = 1/sqrt(x^2+1)
    if (ns == ns_asinh) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(x2, one, ns_add);
        x2 = make_unary(x2, ns_sqrt);
        return make_binary(dout, x2, ns_divide);
    }
    // dacosh(x) = 1/sqrt(x^2-1)
    if (ns == ns_acosh) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(x2, one, ns_subtract);
        x2 = make_unary(x2, ns_sqrt);
        return make_binary(dout, x2, ns_divide);
    }
    // datanh(x) = 1/(1-x^2)
    if (ns == ns_atanh) {
        auto one = make_number(1, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_binary(one, x2, ns_subtract);
        return make_binary(dout, x2, ns_divide);
    }
    // dsigmoid(x) = sigmoid(x) - sigmoid(x)^2
    if (ns == ns_sigmoid) {
        auto r = make_binary(out, out, ns_multiply);
        r = make_binary(out, r, ns_subtract);
        return make_binary(dout, r, ns_multiply);
    }
    return nullptr;
}

void UnaryOp::infer_shape() {
    y->set_shape(x->shape);
}

void UnaryOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("OP", ns.to_cstring());
}

#else // JIT
void UnaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    index_t num = y->num;
    for (index_t i=0; i<num; i++)
        yp[i] = @expand_macro(@OP, Ty, xp[i]);
}
#endif // JIT

} // jittor