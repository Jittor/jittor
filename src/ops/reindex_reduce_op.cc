// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include "var.h"
#include "ops/reindex_reduce_op.h"
#include "ops/binary_op_defs.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_reindex = get_op_info("reindex")
    .get_constructor<VarPtr, Var*, NanoVector, vector<string>&&, float64, vector<string>&&, vector<Var*>&&>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();


ReindexReduceOp::ReindexReduceOp(Var* y, NanoString op, NanoVector shape, vector<string>&& indexes, vector<string>&& overflow_conditions, vector<Var*>&& extras)
    : y(y), shape(shape), indexes(move(indexes)), overflow_conditions(move(overflow_conditions)), extras(extras) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::reduce);
    ns = op;
    ASSERT(ns.is_binary() && ns!=ns_mean);
    x = create_output(nullptr, y->dtype());
}

VarPtr ReindexReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Do not have grad to extras input
    if (v_index) return nullptr;
    if (ns == ns_add)
        return make_reindex(dout, v->shape, clone(indexes), 0, clone(overflow_conditions), move(extras));
    if (ns == ns_multiply) {
        VarPtr a = make_binary(dout, out, ns_multiply);
        VarPtr b = make_reindex(a, v->shape, clone(indexes), 0, clone(overflow_conditions), move(extras));
        return make_binary(b, v, ns_divide);
    }
    if (ns == ns_maximum || ns == ns_minimum) {
        VarPtr zeros = make_number(0, v);
        VarPtr a = make_reindex(out, v->shape, clone(indexes), 0, clone(overflow_conditions), move(extras));
        VarPtr cond = make_binary(v, a, ns_equal);
        VarPtr dv = make_reindex(dout, v->shape, clone(indexes), 0, clone(overflow_conditions), move(extras));
        return make_ternary(cond, dv, zeros);
    }
    return nullptr;
}

void ReindexReduceOp::infer_shape() {
    CHECKop(shape.size(),==,indexes.size()) << "Number of shape and indexes should be the same.";
    CHECK(shape.size()) << "Number of shape should greater than 0.";
    for (auto v : shape)
        CHECKop(v,>=,0u) << "Shape should greater than 0.";
    x->set_shape(shape);
    CHECKop(x->size,>=,0u);
    CHECKop(y->size,>=,0u);
}

void ReindexReduceOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("OP", ns.to_cstring());
    add_jit_define("YDIM", JK::hex1(y->shape.size()));
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
    for (uint i=0; i<indexes.size(); i++)
        add_jit_define("INDEX", JK::hex1(i), indexes[i]);
    add_jit_define("OSIZE", JK::hex1(overflow_conditions.size()));
    for (uint i=0; i<overflow_conditions.size(); i++)
        add_jit_define("OFD", JK::hex1(i), overflow_conditions[i]);
    add_jit_define("ESIZE", JK::hex1(extras.size()));
    for (uint i=0; i<extras.size(); i++) {
        add_jit_define("EDIM", JK::hex1(i), JK::hex1(extras[i]->shape.size()));
        add_jit_define("Te", JK::hex1(i), extras[i]->dtype());
    }
}

#else // JIT
void ReindexReduceOp::jit_run() {
    auto* __restrict__ yp = y->ptr<Tx>();
    // define extra
    @for(i, 0, ESIZE,
        auto* __restrict__ extras@i@@p = extras[@i]->ptr<Te@i>();
        @for(j, 0, EDIM@i, index_t extras@i@@shape@j = extras[@i]->shape[@j];)
        index_t extras@i@@stride@{EDIM@i-1} = 1;
        @for(j, EDIM@i-2, -1, -1, auto extras@i@@stride@j = extras@i@@stride@{j+1} * extras@i@@shape@{j+1};)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    // define x shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define x stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    // define y shape
    @for(i, 0, YDIM, index_t yshape@i = y->shape[@i];)
    // define y stride
    index_t ystride@{YDIM-1} = 1;
    @for(i, YDIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    // init

    @for(d, 0, XDIM, for (index_t i@d=0; i@d < xshape@d; i@d++)) {
        auto xid = @for(d, 0, XDIM, + i@d * xstride@d);
        xp[xid] = @expand_macro(init_@OP, Tx);
    }
    // generate d-for loop
    @for(d, 0, YDIM, for (index_t i@d=0; i@d < yshape@d; i@d++)) {
        auto yid = @for(d, 0, YDIM, + i@d * ystride@d);
        @for(d, 0, XDIM, index_t xid@d = @expand_macro(INDEX@d);)
        auto xid = @for(d, 0, XDIM, + xid@d * xstride@d);
        bool check_overflow = 0 @for(d, 0, XDIM, || xid@d<0 || xid@d>=xshape@d) @for(d, 0, OSIZE, || (@expand_macro(OFD@d)));
        if (!check_overflow)
            xp[xid] = @expand_macro(@OP, Tx, xp[xid], yp[yid]);
    }
}
#endif // JIT

} // jittor