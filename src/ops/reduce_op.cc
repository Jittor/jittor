// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include "var.h"
#include "ops/reduce_op.h"
#include "ops/binary_op_defs.h"
#include "ops/op_register.h"
#include "executor.h"

namespace jittor {

#ifndef JIT
static auto make_broadcast_to = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, Var*, uint>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();

NanoString binary_dtype_infer(NanoString op, Var* dx, Var* dy);

unordered_set<string> reduce_ops = {
    // @pybind(max, reduce_maximum)
    "maximum", 
    // @pybind(min, reduce_minimum)
    "minimum", 
    // @pybind(sum, reduce_add)
    "add",
    // @pybind(prod, product, reduce_multiply)
    "multiply", 
    "logical_and", 
    "logical_or", 
    "logical_xor", 
    "bitwise_and", 
    "bitwise_or", 
    "bitwise_xor",
    // @pybind(mean)
    "mean",
};

ReduceOp::ReduceOp(Var* x, NanoString op, NanoVector dims, bool keepdims)
    : x(x), keepdims(keepdims) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::reduce);
    ns = op;
    ASSERT(ns.is_binary());
    auto xdim = x->shape.size();
    if (!dims.size()) {
        reduce_mask = (1<<xdim)-1;
    } else {
        reduce_mask = 0;
        for (auto dim : dims) {
            CHECKop(dim,<,xdim) << "Wrong dims number:" << dims;
            reduce_mask |= 1<<dim;
        }
    }
    y = create_output(nullptr, binary_dtype_infer(ns, x, x));
}

ReduceOp::ReduceOp(Var* x, NanoString op, uint dims_mask, bool keepdims)
    : x(x), keepdims(keepdims) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::reduce);
    ns = op;
    ASSERT(ns.is_binary());
    auto xdim = x->shape.size();
    reduce_mask = dims_mask ? dims_mask : (1<<xdim)-1;
    y = create_output(nullptr, binary_dtype_infer(ns, x, x));
}

ReduceOp::ReduceOp(Var* x, NanoString op, int dim, bool keepdims)
    : ReduceOp(x, op, NanoVector(dim), keepdims) {}

void ReduceOp::infer_shape() {
    auto xdim = x->shape.size();
    yshape.clear();
    for (int i=0; i<xdim; i++)
        yshape.push_back(((reduce_mask>>i)&1) ? 1 : x->shape[i]);
    if (!keepdims) {
        NanoVector yshape2;
        for (size_t i=0; i<xdim; i++)
            if (!(reduce_mask>>i & 1))
                yshape2.push_back(yshape[i]);
        if (!yshape2.size()) yshape2.push_back(1);
        y->set_shape(yshape2);
    } else
        y->set_shape(yshape);
}

VarPtr ReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    uint mask=0;
    if (!keepdims) mask = reduce_mask;
    if (ns == ns_add)
        return make_broadcast_to(dout, v, mask);
    if (ns == ns_multiply) {
        VarPtr a = make_binary(dout, out, ns_multiply);
        VarPtr b = make_broadcast_to(a, v, mask);
        return make_binary(b, v, ns_divide);
    }
    if (ns == ns_mean) {
        if (v->num < 0) {
            // TODO: Dynamic shape of mean grad was not supported yet
            LOGw << "Dynamic shape of mean grad cause synchronize.";
            exe.run_sync({v}, 0);
            ASSERT(v->num>=0);
        }
        VarPtr a = make_broadcast_to(dout, v, mask);
        VarPtr n = make_number(1.0f*out->num / v->num, a);
        return make_binary(a, n, ns_multiply);
    }
    if (ns == ns_maximum || ns == ns_minimum) {
        VarPtr zeros = make_number(0, v);
        VarPtr a = make_broadcast_to(out, v, mask);
        VarPtr cond = make_binary(v, a, ns_equal);
        VarPtr dv = make_broadcast_to(dout, v, mask);
        return make_ternary(cond, dv, zeros);
    }
    return nullptr;
}

void ReduceOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("Tz", y->dtype());
    add_jit_define("OP", ns.to_cstring());
    add_jit_define("DIM", JK::hex1(yshape.size()));
    add_jit_define("REDUCE", JK::hex(reduce_mask));
}

#else // JIT
void ReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    
    @for(i, 0, DIM, index_t xshape@i = x->shape[@i];)
    @for(i, 0, DIM, index_t yshape@i = @if(REDUCE>>i&1,1,xshape@i);)
    index_t ystride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    Ty count = Ty(x->num) / Ty(y->num);
    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
        yp[yid] = @expand_macro(init_@OP, Ty);
    }
    
    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        @for(d, 0, DIM,@if(REDUCE>>d&1, for (index_t xi@d=0; xi@d < xshape@d; xi@d++),)) {
            auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
            auto xid = 0 @for(d, 0, DIM, + xi@d * xstride@d);
            yp[yid] = @expand_macro(@OP, Ty, yp[yid], xp[xid]);
        }
    }
    (void)count, (void)yshape0, (void)ystride0;
}
#endif // JIT

} // jittor