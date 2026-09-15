// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/fuse_transpose_op.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = op_constructor<VarPtr, Var*, NanoVector>("fuse_transpose");

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
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(tp);
    set_flag(OpFlags::_manual_set_vnbb);
    int i=0;
    for (; i<axes.size(); i++)
        if (i!=axes[i]) break;
    // Only a *complete* identity permutation forwards x. Matching axes[i]==i
    // without also matching the rank made every ascending prefix look like the
    // identity: `axes=[0]` on a 2-D var, or `[0,1,2]` on a 2-D var, returned x
    // unchanged and never reached infer_shape, so the axes checks there could
    // not fire and a wrong `axes` silently did nothing.
    if (i==axes.size() && axes.size() && axes.size()==x->shape.size()) {
        forward(x);
        return;
    }
    auto xdim = x->shape.size();
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    }
    y = create_output(nullptr, x->dtype());
    // ParallelPass hands threads to the outermost `max_parallel_depth` loops,
    // 4 by default on CUDA. A permute of rank 5 or more therefore leaves its
    // innermost loop -- the contiguous one -- running serially inside each
    // thread, so neighbouring threads land a whole inner row apart and not one
    // access in a warp coalesces. The qkv permute of an attention block is
    // exactly that shape, and it measured 152 GB/s against 404 GB/s once the
    // inner loop is parallelized too (166 us -> 62 us at b8s256 d512).
    // MiniMax-H3's decoder hits the same wall from the other side: it folds
    // its (1, T, H, W, C, pt, ph, pw) patch grid back to
    // (1, C, T*pt, H*ph, W*pw) and permutes the 8-D view first, where the
    // outer axes are 1, 3, 7 and 4 -- the whole budget goes to those and a
    // single block of 32 threads moves five million elements.
    //
    // Only for the broadcast form: that one is a pure permute, so every loop
    // is independent and giving them all threads costs nothing. The reduce
    // form fuses into a kernel that may carry a reduction, where parallelizing
    // the reduced axis would mean atomics and a float summation order that
    // changes run to run.
    if (tp == OpType::broadcast && axes.size() > 4) {
        loop_options_t options = y->loop_options;
        options["max_parallel_depth"] = (int)axes.size();
        y->loop_options = move(options);
    }
}

void FuseTransposeOp::infer_shape() {
    auto xdim = x->shape.size();
    USER_CHECK(xdim);
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    } else {
        USER_CHECKop(axes.size(),==,xdim);
        int64_t mask=0;
        for (auto i : axes) mask |= 1<<i;
        USER_CHECK(mask==((1ll<<xdim)-1)) << "Invalid axes" << axes;
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
    jk << "«Tx:" << x->dtype();
    jk << "«DIM=" << JK::hex1(axes.size());
    jk << "«BC:" << JK::hex1(bc);
    for (uint i=0; i<ax.size(); i++)
        jk << "«AXES" << JK::hex1(ax[i]) << '=' << JK::hex1(i);
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
