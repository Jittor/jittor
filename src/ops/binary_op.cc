// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/binary_op_defs.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_broadcast_to = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, Var*, NanoVector>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_unary = get_op_info("unary")
    .get_constructor<VarPtr, Var*, NanoString>();
static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();

unordered_set<string> binary_ops = {
    /**
    Computes ``x^y``, element-wise. 

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(pow, __pow__)
    "pow",

    /**
    Returns the element-wise maximum of ``x`` and ``y``. 

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    "maximum",

    /**
    Returns the element-wise minimum of ``x`` and ``y``. 

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    "minimum",

    /**
    Element-wise adds ``x`` and ``y`` and returns a new Var. 
    
    This operation is equivalent to ``x + y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(add, __add__)
    "add",

    /**
    Element-wise subtract ``y`` from ``x`` and returns a new Var.

    This operation is equivalent to ``x - y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(subtract, __sub__)
    "subtract",

    /**
    Element-wise muliplies ``x`` with ``y`` and returns a new Var.

    This operation is equivalent to ``x * y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(multiply, __mul__)
    "multiply",

    /**
    Element-wise divide ``x`` by ``y`` and returns a new Var.

    This operation is equivalent to ``x / y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

    ----------------

    Example-1::
        >>> a = jt.empty((3,), dtype=jt.int32)
        >>> a
        jt.Var([707406378 707406378 707406378], dtype=int32)
        >>> b = jt.empty((3,), dtype=jt.int32)
        >>> b
        jt.Var([674510453 171649398 538976288], dtype=int32)
        >>> jt.divide(a, b)
        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)
        >>> a / b
        jt.Var([1.0487701 4.1212287 1.3125001], dtype=float32)

    .. note ::
    returns float value even if the dtype of input Vars are both integers.
    @see jt.ops.floor_divide() for floor division.
     */
    // @pybind(divide, __truediv__)
    "divide",

    /**
    Element-wise divide ``x`` by ``y`` and returns the floor of the result.

    This operation is equivalent to ``x // y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randint(1, 10, (3,), dtype=jt.int32)
        >>> a
        jt.Var([9 2 7], dtype=int32)
        >>> b = jt.randint(1, 10, (3,), dtype=jt.int32)
        >>> b
        jt.Var([6 4 6], dtype=int32)
        >>> jt.floor_divide(a, b)
        jt.Var([1 0 1], dtype=int32)
        >>> a // b
        jt.Var([1 0 1], dtype=int32)
     */
    // @pybind(floor_divide, __floordiv__)
    "floor_divide",

    /**
    Returns the element-wise remainder of division.

    This operation is equivalent to ``x % y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(3)
        >>> a
        jt.Var([0.3989529  0.20159635 0.22973768], dtype=float32)
        >>> b = jt.rand(3)
        >>> b
        jt.Var([0.20121202 0.7704864  0.5654395 ], dtype=float32)
        >>> jt.mod(a, b)
        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)
        >>> a % b
        jt.Var([0.19774088 0.20159635 0.22973768], dtype=float32)
     */
    // @pybind(mod, __mod__)
    "mod",

    /**
    Returns ``x < y`` element-wise.

    This operation is equivalent to ``x < y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(less, __lt__)
    "less",

    /**
    Returns ``x <= y`` element-wise.

    This operation is equivalent to ``x <= y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(less_equal, __le__)
    "less_equal",

    /**
    Returns ``x > y`` element-wise.

    This operation is equivalent to ``x > y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(greater, __gt__)
    "greater",

    /**
    Returns ``x >= y`` element-wise.
    
    This operation is equivalent to ``x >= y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(greater_equal, __ge__)
    "greater_equal",

    /**
    Returns ``x == y`` element-wise.

    This operation is equivalent to ``x == y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(equal, __eq__)
    "equal",

    /**
    Returns ``x != y`` element-wise.

    This operation is equivalent to ``x != y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var.

    * [in] y: the second input, a python number or jt.Var.

     */
    // @pybind(not_equal, __ne__)
    "not_equal",

    /**
    Shifts the bits of ``x`` to the left by ``y``. 

    Bits are shifted to the left by appending ``y`` 0s at the right of ``x``.
    This operation is equivalent to ``x << y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var (int32 or int64).

    * [in] y: the second input, a python number or jt.Var (int32 or int64).

    ----------------

    Example-1::
        >>> a = jt.randint(0, 10, shape=(3,))
        >>> a
        jt.Var([7 6 7], dtype=int32)
        >>> b = jt.randint(0, 10, shape=(3,))
        >>> b
        jt.Var([3 9 8], dtype=int32)
        >>> jt.left_shift(a, b)
        jt.Var([  56 3072 1792], dtype=int32)
        >>> a << b
        jt.Var([  56 3072 1792], dtype=int32)
     */
    // @pybind(left_shift, __lshift__)
    "left_shift",

    /**
    Shifts the bits of ``x`` to the right by ``y``. 

    This operation is equivalent to ``x >> y``.

    ----------------

    * [in] x: the first input,  a python number or jt.Var (int32 or int64).

    * [in] y: the second input, a python number or jt.Var (int32 or int64).

    ----------------

    Example-1::
        >>> a = jt.randint(0, 1024, shape=(3,))
        >>> a
        jt.Var([439 113  92], dtype=int32)
        >>> b = jt.randint(0, 10, shape=(3,))
        >>> b
        jt.Var([6 8 4], dtype=int32)
        >>> jt.right_shift(a, b)
        jt.Var([6 0 5], dtype=int32)
     */
    // @pybind(right_shift, __rshift__)
    "right_shift",

    /**
    Returns the element-wise logical AND of the inputs. 

    ----------------

    * [in] x: the first input, jt.Var.

    * [in] y: the second input, jt.Var.

     */
    "logical_and",

    /**
    Returns the element-wise logical OR of the inputs. 

    ----------------

    * [in] x: the first input, jt.Var.

    * [in] y: the second input, jt.Var.

     */
    "logical_or",

    /**
    Returns the element-wise logical XOR of the inputs. 

    ----------------

    * [in] x: the first input, jt.Var.

    * [in] y: the second input, jt.Var.

     */
    "logical_xor",

    /**
    Computes the bitwise AND of x and y.

    ----------------

    * [in] x: the first input, jt.Var (integal or boolean).

    * [in] y: the second input, jt.Var (integal or boolean).

     */
    // @pybind(bitwise_and, __and__)
    "bitwise_and",

    /**
    Computes the bitwise OR of x and y.

    ----------------

    * [in] x: the first input, jt.Var (integal or boolean).

    * [in] y: the second input, jt.Var (integal or boolean).

     */
    // @pybind(bitwise_or, __or__)
    "bitwise_or",

    /**
    Computes the bitwise XOR of x and y.

    ----------------

    * [in] x: the first input, jt.Var (integal or boolean).

    * [in] y: the second input, jt.Var (integal or boolean).

     */
    // @pybind(bitwise_xor, __xor__)
    "bitwise_xor",
};

NanoString binary_dtype_infer(NanoString op, Var* x, Var* y) {
    if (op == ns_mean) return dtype_infer(x->ns, y->ns, 2); // force float
    int force_type=0;
    if (op == ns_divide) force_type=2; // force float
    if (op == ns_floor_divide) force_type=1; // force int
    return op.is_bool() ? ns_bool : dtype_infer(x->ns, y->ns, force_type);
}

BinaryOp::BinaryOp(Var* x, Var* y, NanoString op) : x(x), y(y) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    ns = op;
    ASSERT(ns.is_binary());
    z = create_output(nullptr, binary_dtype_infer(op, x, y));
}

VarPtr dirty_clone_broadcast(Var* v) {
    Op* op = v->input();
    // dirty fix conv duplicated
    if (op && !v->is_finished() && v->shape.size() > 4 && op->type() == OpType::broadcast) {
        auto vp = op->duplicate();
        if (vp) {
            // TODO: loop options should be set to op, rather than var
            if (v->loop_options)
                vp->loop_options = v->loop_options;
            return vp;
        }
    }
    return v;
}

VarPtr BinaryOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (ns == ns_add) return dout;
    if (ns == ns_subtract) {
        if (v_index == 0) 
            return dout;
        else
            return make_unary(dout, ns_negative);
    }
    if (ns == ns_multiply) {
        if (v_index == 0) 
            return make_binary(dirty_clone_broadcast(y), dirty_clone_broadcast(dout), ns_multiply);
        else
            return make_binary(dirty_clone_broadcast(x), dirty_clone_broadcast(dout), ns_multiply);
    }
    if (ns == ns_divide) {
        if (v_index == 0) 
            return make_binary(dout, y, ns_divide);
        else {
            // dy = -dz*x / y^2
            auto ndz = make_unary(dout, ns_negative);
            auto ndzx = make_binary(ndz, x, ns_multiply);
            auto y2 = make_binary(y, y, ns_multiply);
            return make_binary(ndzx, y2, ns_divide);
        }
    }
    if (ns == ns_maximum || ns == ns_minimum) {
        auto zeros = make_number(0, dout);
        auto cond = make_binary(y, z, ns_equal);
        if (v_index==1)
            return make_ternary(cond, dout, zeros);
        else
            return make_ternary(cond, zeros, dout);
    }
    if (ns == ns_pow) {
        if (v_index == 0) {
            // dout * y * x^(y-1)
            auto d = make_binary(dout, y, ns_multiply);
            // auto ones = make_number(1, dout);
            int number = 1;
            auto ones = make_array(&number, 1, ns_int32);
            auto y_1 = make_binary(y, ones, ns_subtract);
            auto x_y_1 = make_binary(x, y_1, ns_pow);
            return make_binary(d, x_y_1, ns_multiply);
        } else {
            // dout * x^y * log(x)
            auto log_x = make_unary(x, ns_log);
            auto x_y_log_x = make_binary(z, log_x, ns_multiply);
            return make_binary(dout, x_y_log_x, ns_multiply);
        }
    }
    return nullptr;
}

void BinaryOp::infer_shape() {
    auto xdim = x->shape.size();
    auto ydim = y->shape.size();
    bool need_broadcast = xdim != ydim;
    for (size_t i=0; i<xdim && i<ydim; i++) {
        auto xshape = x->shape[xdim-i-1];
        auto yshape = y->shape[ydim-i-1];
        // -1 1 need b
        // has 1, b, both 1, not b, 0, error
        if ((xshape == 1 || yshape == 1) && (xshape != yshape)) {
            // CHECK(xshape && yshape) << "Shape can not broadcast to 0.";
            need_broadcast = true;
            continue;
        }
        if (xshape<0 || yshape<0 ) continue;
        CHECKop(xshape,==,yshape) << "Shape not match, x:" >> x->to_string()
            << " y:" >> y->to_string();
    }
    if (need_broadcast) {
        auto xp = make_broadcast_to(x, y, {});
        auto yp = make_broadcast_to(y, x, {});
        set_inputs({x=xp, y=yp});
        // infer shape again
        infer_shape();
    } else
        z->set_shape(x->shape);
}

void BinaryOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype()
        << _CS("][Ty:") << y->dtype()
        << _CS("][Tz:") << z->dtype()
        << _CS("][OP:") << ns << ']';
}

#else // JIT
void BinaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    for (index_t i=0; i<num; i++)
        zp[i] = @expand_macro(@OP, Tz, xp[i], yp[i]);
}
#endif // JIT

} // jittor
