// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/fuse_transpose_op.h"
#include "var.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = get_op_info("fuse_transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();

static inline NanoVector get_reverse(NanoVector axes) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return reverse;
}

FuseTransposeOp::FuseTransposeOp(Var* x, NanoVector axes_) : x(x), axes(axes_) {
    OpType tp = OpType::broadcast;
    if (!x->is_finished()) {
        auto type = x->input()->type();
        if (type==OpType::broadcast || type==OpType::element)
            tp = OpType::reduce;
    }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(tp);
    int i=0;
    for (; i<axes.size(); i++)
        if (i!=axes[i]) break;
    if (i==axes.size() && axes.size()) {
        forward(x);
        return;
    }
    auto xdim = x->shape.size();
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    }
    y = create_output(nullptr, x->dtype());
}

void FuseTransposeOp::infer_shape() {
    auto xdim = x->shape.size();
    CHECK(xdim);
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    } else {
        CHECKop(axes.size(),==,xdim);
        int64_t mask=0;
        for (auto i : axes) mask |= 1<<i;
        CHECK(mask==((1ll<<xdim)-1)) << "Invalid axes" << axes;
    }
    NanoVector shape;
    for (uint i=0; i<xdim; i++)
        shape.push_back(x->shape[axes[i]]);
    y->set_shape(shape);
}

VarPtr FuseTransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_transpose(dout, get_reverse(axes));
}

void FuseTransposeOp::jit_prepare(JK& jk) {
    auto bc = type()==OpType::broadcast;
    auto ax = bc ? axes : get_reverse(axes);
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][DIM=") << JK::hex1(axes.size());
    jk << _CS("][BC:") << JK::hex1(bc);
    for (uint i=0; i<ax.size(); i++)
        jk << _CS("][AXES") << JK::hex1(ax[i]) << '=' << JK::hex1(i);
    jk << ']';
}

#else // JIT
void FuseTransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    
    @for(i, 0, DIM, index_t yshape@i = y->shape[@i];)
    @for(i, 0, DIM, index_t xshape@i = yshape@{AXES@i};)
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    index_t ystride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    @if(BC,
    @for(d, 0, DIM, for (index_t i@d=0; i@d < yshape@d; i@d++)) {
        auto yid = @for(d, 0, DIM, + i@d * ystride@d);
        @for(d, 0, DIM, auto xid@d = i@{AXES@d};)
        auto xid = @for(d, 0, DIM, + xid@d * xstride@d);
        yp[yid] = xp[xid];
    },
    @for(d, 0, DIM, for (index_t i@d=0; i@d < xshape@d; i@d++)) {
        auto xid = @for(d, 0, DIM, + i@d * xstride@d);
        @for(d, 0, DIM, auto yid@d = i@{AXES@d};)
        auto yid = @for(d, 0, DIM, + yid@d * ystride@d);
        yp[yid] = xp[xid];
    }
    )
    // unused var
    (void)xshape0;
}
#endif // JIT

} // jittor