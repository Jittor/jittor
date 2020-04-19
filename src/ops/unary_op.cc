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
    "float",
    "double",
    "int",
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
    "asin",
    "sinh",
    "asinh",
    "tan",
    "atan",
    "tanh",
    "atanh",
    "cos",
    "acos",
    "cosh",
    "acosh",
};

UnaryOp::UnaryOp(Var* x, NanoString op) : x(x) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    ns = op;
    ASSERT(ns.is_unary() | ns.is_dtype());
    NanoString dtype;
    if (ns.is_dtype()) {
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