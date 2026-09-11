// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include "core/var.h"
#include "ops/reduce_op.h"
#include "ops/op_register.h"
#include "core/executor.h"
#include "runtime/backend.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT
static auto make_broadcast_to = op_constructor<VarPtr, Var*, Var*, uint, uint>("broadcast_to");
static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_reduce = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");
static auto make_reduce2 = op_constructor<VarPtr, Var*, NanoString, uint, uint>("reduce");
static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");

unordered_set<string> reduce_ops = {
    /**
    Returns the maximum elements in the input.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.max(x)
        jt.Var([4], dtype=int32)
        >>> x.max()
        jt.Var([4], dtype=int32)
        >>> x.max(dim=1)
        jt.Var([4 4], dtype=int32)
        >>> x.max(dim=1, keepdims=True)
        jt.Var([[4]
         [4]], dtype=int32)
     */
    // @pybind(max, reduce_maximum)
    "maximum", 

    /**
    Returns the minimum elements in the input.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.min(x)
        jt.Var([0], dtype=int32)
        >>> x.min()
        jt.Var([0], dtype=int32)
        >>> x.min(dim=1)
        jt.Var([1 0], dtype=int32)
        >>> x.min(dim=1, keepdims=True)
        jt.Var([[1]
         [0]], dtype=int32)
     */
    // @pybind(min, reduce_minimum)
    "minimum", 

    /**
    Returns the sum of the input.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[4 1 2]
         [0 2 4]], dtype=int32)
        >>> jt.sum(x)
        jt.Var([13], dtype=int32)
        >>> x.sum()
        jt.Var([13], dtype=int32)
        >>> x.sum(dim=1)
        jt.Var([7 6], dtype=int32)
        >>> x.sum(dim=1, keepdims=True)
        jt.Var([[7]
         [6]], dtype=int32)
     */
    // @pybind(sum, reduce_add)
    "add",

    /**
    Returns the product of all the elements in the input.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[7 5 5]
         [5 7 5]], dtype=int32)
        >>> jt.prod(x)
        jt.Var([30625], dtype=int32)
        >>> x.prod()
        jt.Var([30625], dtype=int32)
        >>> x.prod(dim=1)
        jt.Var([175 175], dtype=int32)
        >>> x.prod(dim=1, keepdims=True)
        jt.Var([[175]
         [175]], dtype=int32)
     */
    // @pybind(prod, product, reduce_multiply)
    "multiply", 

    /**
    Tests if all elements in input evaluate to True.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(2, shape=(2, 3))
        >>> x
        jt.Var([[1 1 1]
         [0 1 0]], dtype=int32)
        >>> jt.all_(x)
        jt.Var([False], dtype=int32)
        >>> x.all_()
        jt.Var([False], dtype=int32)
        >>> x.all_(dim=1)
        jt.Var([True False], dtype=int32)
        >>> x.all_(dim=1, keepdims=True)
        jt.Var([[True]
         [False]], dtype=int32)
     */
    // @pybind(reduce_logical_and, all_)
    "logical_and", 

    /**
    Tests if any elements in input evaluate to True.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(2, shape=(2, 3))
        >>> x
        jt.Var([[1 0 1]
         [0 0 0]], dtype=int32)
        >>> jt.any_(x)
        jt.Var([True], dtype=int32)
        >>> x.any_()
        jt.Var([True], dtype=int32)
        >>> x.any_(dim=1)
        jt.Var([True False], dtype=int32)
        >>> x.any_(dim=1, keepdims=True)
        jt.Var([[True]
         [False]], dtype=int32)
     */
    // @pybind(reduce_logical_or, any_)
    "logical_or", 
    "logical_xor", 
    "bitwise_and", 
    "bitwise_or", 
    "bitwise_xor",

    /**
    Returns the mean value of the input.

    ----------------

    * [in] x:       the input jt.Var.

    * [in] dim or dims:     int or tuples of ints (optional). If specified, reduce along the given the dimension(s).

    * [in] keepdims: bool (optional). Whether the output has ``dim`` retained or not. Defaults to be False.

    ----------------

    Example-1::
        >>> x = jt.randint(10, shape=(2, 3))
        >>> x
        jt.Var([[9 4 4]
         [1 9 6]], dtype=int32)
        >>> jt.mean(x)
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean()
        jt.Var([5.5000005], dtype=float32)
        >>> x.mean(dim=1)
        jt.Var([5.666667  5.3333335], dtype=float32)
        >>> x.mean(dim=1, keepdims=True)
        jt.Var([[5.666667 ]
         [5.3333335]], dtype=float32)
     */
    // @pybind(mean)
    "mean",
};

EXTERN_LIB int amp_reg;

ReduceOp::ReduceOp(Var* x, NanoString op, NanoVector dims, bool keepdims)
    : x(x) {
    const auto& policy = backend_ops(construction_target_backend(x)).execution;
    // improve float16 mean precision
    if (!policy.native_low_precision_reduction && !(amp_reg & 32) && (x->dtype() == ns_float16 || x->dtype() == ns_bfloat16) && (op == ns_mean || op == ns_add)) {
        auto x_float32 = make_unary(x, ns_float32);
        auto mean = make_reduce(x_float32, op, dims, keepdims);
        mean = make_unary(mean, x->dtype());
        forward(mean);
        return;
    }
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::reduce);
    if (op.get(NanoString::_no_need_back_in))
        set_flag(OpFlags::_manual_set_vnbb);
    ns = op;
    USER_CHECK(ns.is_binary()) << "reduce requires a binary reduction operation, got" << ns;
    // The same integral-only rule BinaryOp applies elementwise. Reducing with a
    // bitwise operation over a float reaches a raw `float & float` in the
    // generated kernel, and g++ answers with "invalid operands of types 'float'
    // and 'float'" pointed at `Ty rcount = y->num*1.0 / x->num;` -- a line that
    // has nothing to do with the mistake, because the generated kernel's line
    // numbers do not map back to the template. `ns_is_integral_only` is the
    // predicate BinaryOp uses, so the two paths cannot drift apart.
    if (ns_is_integral_only(ns))
        USER_CHECK(x->dtype().is_int() || x->dtype().is_bool())
            << "Reduce op '" >> ns.to_cstring() >>
            "' requires an integer or boolean dtype, but got x:" >>
            x->dtype().to_cstring() <<
            "(bitwise and shift reductions are not defined for floating-point or complex types).";
    auto xdim = x->shape.size();
    keepdims_mask = keepdims ? (int)-1 : (int)0;
    if (!dims.size()) {
        reduce_mask = (1<<xdim)-1;
    } else {
        reduce_mask = 0;
        for (auto dim : dims) {
            if (dim<0) dim += xdim;
            USER_CHECK(dim>=0 && dim<xdim) << "Reduce dim out of range: requested dims" << dims
                << "for a" << (int)xdim >> "-D var; valid dims are" << -(int)xdim
                << "to" << ((int)xdim-1) >> ".";
            reduce_mask |= 1<<dim;
        }
    }
    // if (x->dtype() == ns_bool && ns == ns_add)
    if (x->dtype() == ns_bool)
        y = create_output(nullptr, ns_int32);
    else
        y = create_output(nullptr, reduce_dtype_infer(ns, x->ns, policy.preserve_reduction_dtype));
}

ReduceOp::ReduceOp(Var* x, NanoString op, uint dims_mask, uint keepdims_mask)
    : x(x) {
    const auto& policy = backend_ops(construction_target_backend(x)).execution;
    // improve float16 mean precision
    if (!policy.native_low_precision_reduction && !(amp_reg & 32) && (x->dtype() == ns_float16 || x->dtype() == ns_bfloat16) && (op == ns_mean || op == ns_add)) {
        auto x_float32 = make_unary(x, ns_float32);
        auto mean = make_reduce2(x_float32, op, dims_mask, keepdims_mask);
        mean = make_unary(mean, x->dtype());
        forward(mean);
        return;
    }
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::reduce);
    if (op.get(NanoString::_no_need_back_in))
        set_flag(OpFlags::_manual_set_vnbb);
    ns = op;
    USER_CHECK(ns.is_binary()) << "reduce requires a binary reduction operation, got" << ns;
    // The same integral-only rule BinaryOp applies elementwise. Reducing with a
    // bitwise operation over a float reaches a raw `float & float` in the
    // generated kernel, and g++ answers with "invalid operands of types 'float'
    // and 'float'" pointed at `Ty rcount = y->num*1.0 / x->num;` -- a line that
    // has nothing to do with the mistake, because the generated kernel's line
    // numbers do not map back to the template. `ns_is_integral_only` is the
    // predicate BinaryOp uses, so the two paths cannot drift apart.
    if (ns_is_integral_only(ns))
        USER_CHECK(x->dtype().is_int() || x->dtype().is_bool())
            << "Reduce op '" >> ns.to_cstring() >>
            "' requires an integer or boolean dtype, but got x:" >>
            x->dtype().to_cstring() <<
            "(bitwise and shift reductions are not defined for floating-point or complex types).";
    reduce_mask = dims_mask;
    this->keepdims_mask = keepdims_mask;
    y = create_output(nullptr, reduce_dtype_infer(ns, x->ns, policy.preserve_reduction_dtype));
}

ReduceOp::ReduceOp(Var* x, NanoString op, int dim, bool keepdims)
    : ReduceOp(x, op, NanoVector(dim), keepdims) {}

void ReduceOp::infer_shape() {
    auto xdim = x->shape.size();
    // A max or min over zero elements has no answer. add and multiply have
    // identities (0 and 1) and mean of nothing is nan, so those reductions of
    // an empty var are well defined and stay legal; maximum and minimum have
    // none, and the kernel's seed -- the dtype's lowest/highest finite value --
    // was being returned as if it were data, so `jt.zeros((0,3)).max(0)`
    // answered [-3.4e38, -3.4e38, -3.4e38]. numpy raises ValueError for this
    // reduction and torch raises RuntimeError; USER_CHECK makes it the same
    // kind of catchable caller error here.
    if (ns == ns_maximum || ns == ns_minimum)
        for (uint i=0; i<xdim; i++)
            USER_CHECK(!((reduce_mask>>i&1) && x->shape[i]==0))
                << "Reduce" << ns >> ": dim" << i << "of x" << x->shape
                << "is empty, and" << ns
                << "has no identity to return over zero elements.";
    NanoVector yshape; 
    yshape.clear();
    for (int i=0; i<xdim; i++) {
        if (reduce_mask>>i&1) {
            if (keepdims_mask>>i&1)
                yshape.push_back(1);
        } else
            yshape.push_back(x->shape[i]);
    }
    y->set_shape(yshape);
    if (yshape.size() == 0)
        y->set_flag(VarFlags::_is_scalar);
}

VarPtr ReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (ns == ns_add) {
        auto ret = make_broadcast_to(dout, v, reduce_mask, keepdims_mask);
        return ret;
    }
    if (ns == ns_multiply) {
        VarPtr a = make_binary(dout, out, ns_multiply);
        VarPtr b = make_broadcast_to(a, v, reduce_mask, keepdims_mask);
        return make_binary(b, v, ns_divide);
    }
    if (ns == ns_mean) {
        VarPtr a = make_broadcast_to(dout, v, reduce_mask, keepdims_mask);
        VarPtr n = make_number(1.0f*out->num / v->num, a);
        return make_binary(a, n, ns_multiply);
    }
    if (ns == ns_maximum || ns == ns_minimum) {
        // Every element equal to the extremum is on the mask, so handing each
        // of them the whole cotangent multiplies the gradient by the number of
        // ties: max([1,3,3]) produced [0,1,1], which sums to 2. A subgradient
        // of a 1-homogeneous selection has to sum to 1 whichever tie-breaking
        // rule is chosen, so this splits it evenly -- the rule torch's `amax`
        // uses, and the one that matches this op's shape (it returns values,
        // not the (values, indices) pair `torch.max(dim)` returns).
        //
        // The extra reduction counts the ties. With no ties it divides by one,
        // which is the common case and the reason the cost is acceptable here:
        // this is the backward, which already broadcasts twice.
        VarPtr zeros = make_number(0, v);
        VarPtr ones = make_number(1, v);
        VarPtr a = make_broadcast_to(out, v, reduce_mask, keepdims_mask);
        VarPtr cond = make_binary(v, a, ns_equal);
        VarPtr hits = make_ternary(cond, ones, zeros);
        VarPtr n = make_reduce2(hits, ns_add, reduce_mask, keepdims_mask);
        VarPtr nb = make_broadcast_to(n, v, reduce_mask, keepdims_mask);
        VarPtr dv = make_broadcast_to(dout, v, reduce_mask, keepdims_mask);
        VarPtr share = make_binary(dv, nb, ns_divide);
        return make_ternary(cond, share, zeros);
    }
    return nullptr;
}

void ReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype()
        << "«Ty:" << y->dtype()
        << "«Tz:" << y->dtype()
        << "«OP:" << ns
        << "«DIM=" << JK::hex1(x->shape.size())
        << "«REDUCE=" << JK::hex(reduce_mask)
        << "«EMPTY_MEAN=" << JK::hex1(x->num == 0 && ns == ns_mean);
}

#else // JIT
void ReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    
    @for(i, 0, DIM, index_t xshape@i = x->shape[@i];)
    @for(i, 0, DIM, index_t yshape@i = @if(REDUCE>>i&1,1,xshape@i);)
    // A rank-0 input reduces over no dimensions at all, and `@{DIM-1}` is
    // then `-1`: this line used to emit `index_t ystride-1 = 1;`, which does
    // not compile. `loss.sum()` where the loss is already a scalar is ordinary
    // code -- PyTorch returns the value unchanged -- and it died here on both
    // devices (KI-OPS-004). With the guard, every `@for` below produces an
    // empty nest, the body runs once with `yid == xid == 0`, and the result is
    // the input value, which is what the reduction of a single element is.
    @if(DIM>0, index_t ystride@{DIM-1} = 1;)
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    @for(i, 0, DIM, index_t xstride@i = x->storage_stride(@i);)
    Ty count = x->num*1.0 / y->num;
    Ty rcount = y->num*1.0 / x->num;
    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
        yp[yid] = @expand_op(init_@OP, @Ty);
        @if(@EMPTY_MEAN,
            yp[yid] = (Ty)::nanf("");
            ,
        )
    }
    
    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        @for(d, 0, DIM,@if(REDUCE>>d&1, for (index_t xi@d=0; xi@d < xshape@d; xi@d++),)) {
            auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
            auto xid = 0 @for(d, 0, DIM, + xi@d * xstride@d);
            yp[yid] = @expand_op(@OP, @Ty, yp[yid], @Ty, xp[xid], @Tx);
        }
    }
    (void)count; (void)rcount;
    @if(DIM>0, (void)yshape0; (void)ystride0;)
}
#endif // JIT

} // jittor
