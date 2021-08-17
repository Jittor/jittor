// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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

    /**
    Returns the absolute value of the input ``x``. 

    ----------------

    * [in] x:   the input jt.Var

    ----------------
    
    Example-1::
        >>> jt.abs(jt.float32([-1, 0, 1]))
        jt.Var([1. 0. 1.], dtype=float32)
     */
    // @pybind(abs, __abs__)
    "abs",

    /**
    Returns the negative value of the input ``x``. 

    This operator is equavilant to ``-x``.

    ----------------

    * [in] x:   the input jt.Var.

    ----------------
    
    Example-1::
        >>> jt.negative(jt.float32([-1, 0, 1]))
        jt.Var([ 1. -0. -1.], dtype=float32)
     */
    // @pybind(negative, __neg__)
    "negative",

    /**
    Returns the logical NOT of the input ``x``. 
     
    ----------------

    * [in] x: the input jt.Var, integal or boolean.

    ----------------

    Example-1::
        >>> jt.logical_not(jt.int32([-1, 0, 1]))
        jt.Var([False  True False], dtype=bool)
     */
    "logical_not",

    /**
    Returns the bitwise NOT of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var, integal or boolean.

    ----------------

    Example-1::
        >>> jt.bitwise_not(jt.int32([1, 2, -3]))
        jt.Var([-2 -3  2], dtype=int32)
     */
    "bitwise_not",

    /**
    Returns the natural logarithm of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) * 2
        >>> a
        jt.Var([0.02863695 1.30122    1.6048753  1.140261  ], dtype=float32)
        >>> jt.log(a)
        jt.Var([-3.5530574   0.26330233  0.47304606  0.13125724], dtype=float32)
     */
    "log",

    /**
     Returns the exponential of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) * 2
        >>> a
        jt.Var([1.9841381 1.4103996 0.5855549 1.4212812], dtype=float32)
        >>> jt.exp(a)
        jt.Var([7.2727766 4.0975924 1.7959872 4.1424246], dtype=float32)
     */
    "exp",

    /**
    Returns the square root of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) * 2
        >>> a
        jt.Var([0.81957287 0.5609612  0.07435933 1.7571875 ], dtype=float32)
        >>> jt.sqrt(a)
        jt.Var([0.90530264 0.7489734  0.27268907 1.3255895 ], dtype=float32)
     */
    "sqrt",

    /**
    Returns the closest integer of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
        >>> jt.round(a)
        jt.Var([ 2.0  0.0  0.0 -1.0], dtype=float32)
     */
    "round",

    /**
     Returns the largest integer less than or equal to the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------
    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
        >>> jt.floor(a)
        jt.Var([-2.0 -1.0 -1.0 -1.0], dtype=float32)
     */
    "floor",

    /**
    Returns the smallest integer greater than or equal to the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
        >>> jt.ceil(a)
        jt.Var([-1.0  0.0  0.0  0.0], dtype=float32)
     */
    "ceil",

    /**
    Returns the closest integer of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 2.101595    0.33055413 -0.44147047 -0.7720668 ], dtype=float32)
        >>> jt.round_int(a)
        jt.Var([ 2  0  0 -1], dtype=int32)
     */
    "round_int",

    /**
     Returns the largest integer less than or equal to the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------
    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
        >>> jt.floor_int(a)
        jt.Var([-2 -1 -1 -1], dtype=int32)
     */
    "floor_int",

    /**
    Returns the smallest integer greater than or equal to the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-1.0339162 -0.7259972 -0.9220003 -0.8449701], dtype=float32)
        >>> jt.ceil_int(a)
        jt.Var([-1  0  0  0], dtype=int32)
     */
    "ceil_int",

    /**
    Returns the sine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
        >>> jt.sin(a)
        jt.Var([ 0.32303742 -0.6527857  -0.76586854  0.9738172 ], dtype=float32)
     */
    "sin",

    /**
    Returns the arcsine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.09342023 -0.42522037  0.9264933  -0.785264  ], dtype=float32)
        >>> jt.asin(a)
        jt.Var([ 0.09355665 -0.43920535  1.1849847  -0.9031224 ], dtype=float32)
     */
    // @pybind(asin, arcsin)
    "asin",

    /**
    Returns the hyperbolic sine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
        >>> jt.sinh(a)
        jt.Var([ 0.3349012  -0.77276015 -0.9873369   2.9425898 ], dtype=float32)
     */
    "sinh",

    /**
    Returns the inverse hyperbolic sine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-1.9749726  -0.52341473  0.8906148   1.0338128 ], dtype=float32)
        >>> jt.asinh(a)
        jt.Var([-1.4323865  -0.5020559   0.8018747   0.90508187], dtype=float32)
     */
    // @pybind(asinh, arcsinh)
    "asinh",

    /**
    Returns the tangent of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
        >>> jt.tan(a)
        jt.Var([ 0.34133783 -0.8617148  -1.1910915  -4.283673  ], dtype=float32)
     */
    "tan",

    /**
    Returns the inverse tangent of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
        >>> jt.atan(a)
        jt.Var([-0.70961297  0.87102956  0.44140393  0.76464504], dtype=float32)
     */
    // @pybind(atan, arctan)
    "atan",

    /**
    Returns the hyperbolic tangent of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------
    
    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([-0.85885596  1.187804    0.47249675  0.95933187], dtype=float32)
        >>> jt.tanh(a)
        jt.Var([-0.6956678   0.82989657  0.4402144   0.7439787 ], dtype=float32)
     */
    "tanh",

    /**
    Returns the inverse hyperbolic tangent of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) * 2 - 1
        >>> a
        jt.Var([ 0.9062414  -0.799802   -0.27219176 -0.7274077 ], dtype=float32)
        >>> jt.atanh(a)
        jt.Var([ 1.5060828  -1.0980625  -0.27922946 -0.9231999 ], dtype=float32)
     */
    // @pybind(atanh, arctanh)
    "atanh",

    /**
    Returns the cosine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
        >>> jt.cos(a)
        jt.Var([ 0.9463862  0.7575426  0.6429972 -0.2273323], dtype=float32)
     */
    "cos",

    /**
    Returns the inverse cosine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) * 2 - 1
        >>> a
        jt.Var([ 0.5876564  0.740723  -0.667666   0.5371753], dtype=float32)
        >>> jt.acos(a)
        jt.Var([0.9426371 0.7366504 2.3018656 1.0037117], dtype=float32)
     */
    // @pybind(acos, arccos)
    "acos",

    /**
    Returns the hyperbolic cosine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.32893723 -0.7112559  -0.872391    1.8001337 ], dtype=float32)
        >>> jt.cosh(a)
        jt.Var([1.0545894 1.2637873 1.405288  3.1078668], dtype=float32)
     */
    "cosh",

    /**
    Returns the inverse hyperbolic cosine of the input ``x``. 

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.rand(4) + 1
        >>> a
        jt.Var([1.3609099 1.8137748 1.1146184 1.3911307], dtype=float32)
        >>> jt.acosh(a)
        jt.Var([0.8259237  1.2020639  0.47432774 0.8579033 ], dtype=float32)
     */
    // @pybind(acosh, arccosh)
    "acosh",

    /**
    Returns the sigmoid of the input ``x``. 
    
    .. math::
       out_i = \frac{1}{1 + e^{x_i}}

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
        >>> jt.sigmoid(a)
        jt.Var([0.62114954 0.6060032  0.2618374  0.2204857 ], dtype=float32)
     */
    "sigmoid",

    /**
    Computes the error function of each element. The error function is defined as follows:

    .. math::
        erf(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt

    ----------------

    * [in] x: the input jt.Var.

    ----------------

    Example-1::
        >>> a = jt.randn(4)
        >>> a
        jt.Var([ 0.49443012  0.4305426  -1.0364404  -1.2628382 ], dtype=float32)
        >>> jt.erf(a)
        jt.Var([ 0.51559156  0.45739546 -0.85728306 -0.9258883 ], dtype=float32)
     */
    "erf",
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
    // derf(x) = e^(-x^2)*2/sqrt(pi)
    if (ns == ns_erf) {
        auto two_div_sqrt_pi = make_number(2/1.7724538509055159, x);
        auto two = make_number(2, x);
        auto x2 = make_binary(x, x, ns_multiply);
        x2 = make_unary(x2, ns_negative);
        auto r = make_unary(x2, ns_exp);
        r = make_binary(r, two_div_sqrt_pi, ns_multiply);
        return make_binary(dout, r, ns_multiply);
    }
    return nullptr;
}

void UnaryOp::infer_shape() {
    y->set_shape(x->shape);
}

void UnaryOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype()
        << _CS("][Ty:") << y->dtype()
        << _CS("][OP:") << ns << ']';
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