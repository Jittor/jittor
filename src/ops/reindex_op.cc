// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/reindex_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_reindex_reduce = get_op_info("reindex_reduce")
    .get_constructor<VarPtr, Var*, NanoString, NanoVector, vector<string>&&, vector<string>&&, vector<Var*>&&>();
static auto make_reindex = get_op_info("reindex")
    .get_constructor<VarPtr, Var*, NanoVector, vector<string>&&, float64, vector<string>&&, vector<Var*>&&>();
    
ReindexOp::ReindexOp(Var* x, NanoVector shape, vector<string>&& indexes, float64 overflow_value, vector<string>&& overflow_conditions, vector<Var*>&& extras)
    : x(x), 
      shape(shape), 
      indexes(move(indexes)), 
      overflow_conditions(move(overflow_conditions)), 
      overflow_value(overflow_value),
      extras(extras) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    y = create_output(nullptr, x->dtype());
}

ReindexOp::ReindexOp(Var* x, vector<Var*>&& indexes, float64 overflow_value, vector<string>&& overflow_conditions) 
    : x(x), overflow_conditions(move(overflow_conditions)), overflow_value(overflow_value) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    y = create_output(nullptr, x->dtype());
    ASSERTop(indexes.size(),==,x->shape.size());
    auto& shape = indexes[0]->shape;
    ASSERT(indexes.size()<=10 && shape.size()<=10);

    string temp;
    temp.reserve(6+3*shape.size()); // @e0(i0,i1)
    temp += "@e0(";
    for (uint i=0; i<shape.size(); i++) {
        if (i) temp += ',';
        temp += 'i';
        temp += '0'+i;
    };
    temp += ')';
    this->indexes.reserve(indexes.size());
    for (uint i=0; i<indexes.size(); i++) {
        auto& ns = indexes[i]->shape;
        ASSERTop(ns.size(),==,shape.size());
        for (uint j=0; j<ns.size(); j++) ASSERTop(ns[j],==,shape[j]);
        temp[2] = '0'+i; // @ei
        this->indexes.emplace_back(temp);
    }
    // TODO: fix it, we can't move indexes now,
    //     because we need it to add_inputs outside
    // extras = move(indexes);
    extras = indexes;
    for (uint i = 0; i < indexes.size(); ++i) {
        indexes[i]->flags.set(NodeFlags::_force_fuse);
    }
}

VarPtr ReindexOp::duplicate() {
    return make_reindex(x, shape, clone(indexes), overflow_value, clone(overflow_conditions), clone(extras));
}

VarPtr ReindexOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Do not have grad to extras input
    if (v_index) return nullptr;
    return make_reindex_reduce(dout, ns_add, x->shape, clone(indexes), clone(overflow_conditions), move(extras));
}

void ReindexOp::infer_shape() {
    CHECKop(x->shape.size(),==,indexes.size()) << "Number of x's shape and indexes should be the same.";
    if (shape.size())
        y->set_shape(shape);
    else {
        ASSERT(extras.size());
        y->set_shape(extras[0]->shape);
    }
    CHECK(y->shape.size()) << "Number of shape should greater than 0.";
}

void ReindexOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype()
        << _CS("][XDIM=") << JK::hex1(x->shape.size())
        << _CS("][YDIM=") << JK::hex1(y->shape.size())
        << _CS("][OVERFLOW:") << overflow_value;
    for (uint i=0; i<indexes.size(); i++)
        jk << _CS("][INDEX") << JK::hex1(i) << ':' << indexes[i];
    jk << _CS("][OSIZE=") << JK::hex1(overflow_conditions.size());
    for (uint i=0; i<overflow_conditions.size(); i++)
        jk << _CS("][OFD") << JK::hex1(i) << ':' << overflow_conditions[i];
    jk << _CS("][ESIZE=") << JK::hex1(extras.size());
    for (uint i=0; i<extras.size(); i++) {
        jk << _CS("][EDIM") << JK::hex1(i) << '=' << JK::hex1(extras[i]->shape.size());
        jk << _CS("][Te") << JK::hex1(i) << ':' << extras[i]->dtype();
    }
    jk << ']';
}

#else // JIT
void ReindexOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    // define extra
    @for(i, 0, ESIZE,
        auto* __restrict__ extras@i@@p = extras[@i]->ptr<Te@i>();
        @for(j, 0, EDIM@i, index_t extras@i@@shape@j = extras[@i]->shape[@j];)
        index_t extras@i@@stride@{EDIM@i-1} = 1;
        @for(j, EDIM@i-2, -1, -1, auto extras@i@@stride@j = extras@i@@stride@{j+1} * extras@i@@shape@{j+1};)
    )
    auto* __restrict__ yp = y->ptr<Tx>();
    // define y shape
    @for(i, 0, YDIM, index_t yshape@i = y->shape[@i];)
    // define y stride
    index_t ystride@{YDIM-1} = 1;
    @for(i, YDIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    // define x shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define x stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    // generate d-for loop
    @for(d, 0, YDIM, for (index_t i@d=0; i@d < yshape@d; i@d++)) {
        auto yid = @for(d, 0, YDIM, + i@d * ystride@d);
        @for(d, 0, XDIM, index_t xid@d = @expand_macro(INDEX@d);)
        auto xid = @for(d, 0, XDIM, + xid@d * xstride@d);
        bool check_overflow = 0 @for(d, 0, XDIM, || xid@d<0 || xid@d>=xshape@d) @for(d, 0, OSIZE, || (@expand_macro(OFD@d)));
        yp[yid] = check_overflow ? (@OVERFLOW) : xp[xid];
    }
}
#endif // JIT

} // jittor