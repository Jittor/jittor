// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include "var.h"
#include "ops/reindex_reduce_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_reindex = op_constructor<VarPtr, Var*, NanoVector, vector<string>&&, float64, vector<string>&&, vector<Var*>&&>("reindex");
static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");


ReindexReduceOp::ReindexReduceOp(Var* y, NanoString op, NanoVector shape, vector<string>&& indexes, vector<string>&& overflow_conditions, vector<Var*>&& extras)
    : y(y), shape(shape), indexes(move(indexes)), overflow_conditions(move(overflow_conditions)), extras(extras) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::reduce);
    if (op.get(NanoString::_no_need_back_in))
        set_flag(OpFlags::_manual_set_vnbb);
    ns = op;
    ASSERT((ns.is_binary() && ns!=ns_mean) || ns == ns_void);
    x = create_output(nullptr, y->dtype());
    for (auto e : extras) {
        if (e->shape != y->shape) {
            e->set_flag(VarFlags::_stop_fuse);
        }
        if (op.get(NanoString::_no_need_back_in))
            e->set_flag(VarFlags::_needed_by_backward);
    }
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
    USER_CHECKop(shape.size(),==,indexes.size()) << "Number of shape and indexes should be the same.";
    USER_CHECK(shape.size()) << "Number of shape should greater than 0.";
    for (auto v : shape)
        CHECKop(v,>=,0u) << "Shape should greater than 0.";
    x->set_shape(shape);
    CHECKop(x->size,>=,0u);
    CHECKop(y->size,>=,0u);
}

void ReindexReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype()
        << "«OP:" << ns
        << "«YDIM=" << JK::hex1(y->shape.size())
        << "«XDIM=" << JK::hex1(x->shape.size());
    for (uint i=0; i<indexes.size(); i++)
        jk << "«INDEX" << JK::hex1(i) << ':' << indexes[i];
    jk << "«OSIZE=" << JK::hex1(overflow_conditions.size());
    for (uint i=0; i<overflow_conditions.size(); i++)
        jk << "«OFD" << JK::hex1(i) << ':' << overflow_conditions[i];
    jk << "«ESIZE=" << JK::hex1(extras.size());
    for (uint i=0; i<extras.size(); i++) {
        jk << "«EDIM" << JK::hex1(i) << '=' << JK::hex1(extras[i]->shape.size());
        jk << "«Te" << JK::hex1(i) << ':' << extras[i]->dtype();
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

    @if(@strcmp(@OP, void)==0,, 
    @for(d, 0, XDIM, for (index_t i@d=0; i@d < xshape@d; i@d++)) {
        auto xid = @for(d, 0, XDIM, + i@d * xstride@d);
        xp[xid] = @expand_op(init_@OP, @Tx);
    }
    ) // end @if
    
    // generate d-for loop
    @for(d, 0, YDIM, for (index_t i@d=0; i@d < yshape@d; i@d++)) {
        auto yid = @for(d, 0, YDIM, + i@d * ystride@d);
        @for(d, 0, XDIM, index_t xid@d = @expand_macro(INDEX@d);)
        auto xid = @for(d, 0, XDIM, + xid@d * xstride@d);
        bool check_overflow = 0 @for(d, 0, XDIM, || xid@d<0 || xid@d>=xshape@d) @for(d, 0, OSIZE, || (@expand_macro(OFD@d)));
        if (!check_overflow)
            xp[xid] = @expand_op(@OP, @Tx, xp[xid], @Tx, yp[yid], @Tx);
    }
}
#endif // JIT

} // jittor
