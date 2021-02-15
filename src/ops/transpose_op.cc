// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/transpose_op.h"
#include "var.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = get_op_info("transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();

TransposeOp::TransposeOp(Var* x, NanoVector axes_) : x(x), axes(axes_) {
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
    #ifdef HAS_CUDA
    if (use_cuda) {
        static VarPtr(*cutt_transpose)(Var*, NanoVector) = nullptr;
        if (!cutt_transpose && has_op("cutt_transpose")) {
            cutt_transpose = get_op_info("cutt_transpose")
                .get_constructor<VarPtr, Var*, NanoVector>();
        }
        if (cutt_transpose) {
            auto var = cutt_transpose(x, axes);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
}

void TransposeOp::infer_shape() {
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

VarPtr TransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return make_transpose(dout, reverse);
}

void TransposeOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][DIM=") << JK::hex1(axes.size());
    for (uint i=0; i<axes.size(); i++)
        jk << _CS("][AXES") << JK::hex1(axes[i]) << '=' << JK::hex1(i);
    jk << ']';
}

#else // JIT
#ifdef JIT_cpu
void TransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    
    @for(i, 0, DIM, index_t yshape@i = y->shape[@i];)
    @for(i, 0, DIM, index_t xshape@i = yshape@{AXES@i};)
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    index_t ystride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    @for(d, 0, DIM, for (index_t yi@d=0; yi@d < yshape@d; yi@d++)) {
        auto yid = @for(d, 0, DIM, + yi@d * ystride@d);
        @for(d, 0, DIM, auto xi@d = yi@{AXES@d};)
        auto xid = @for(d, 0, DIM, + xi@d * xstride@d);
        yp[yid] = xp[xid];
    }
    // unused var
    (void)xshape0;
}
#else
void TransposeOp::jit_run() {
    // cuda device code
}
#endif // JIT_cpu
#endif // JIT

} // jittor