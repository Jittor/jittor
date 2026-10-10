// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/transpose_op.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "ops/composite/op_capability.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = op_constructor<VarPtr, Var*, NanoVector>("transpose");

TransposeOp::TransposeOp(Var* x, NanoVector axes_) : x(x), axes(axes_) {
    // A rank-0 var has no axes to permute, and the empty permutation of
    // nothing is itself. Both references agree: NumPy's `transpose` returns
    // shape `()`, and torch's `.T` says so outright -- "This function is the
    // identity in these cases". Falling through instead reached
    // `infer_shape`'s `USER_CHECK(xdim)` and made every scalar-shaped
    // `einops.rearrange` die with `transpose_op.cc:61: [check failed: xdim]`.
    // It also read `axes[xdim-1]` below, which is `axes[-1]` when xdim is 0.
    //
    // Only the *empty* permutation forwards: torch rejects `permute((0,))` on
    // a rank-0 tensor, and so should the rank check further down.
    if (!x->shape.size() && !axes.size()) {
        forward(x);
        return;
    }
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
    if (axes.size() < xdim || (axes.size() == xdim && axes[xdim-1]==xdim-1)) {
        static VarPtr(*fuse_transpose)(Var*, NanoVector) = get_op_info("fuse_transpose").get_constructor<VarPtr, Var*, NanoVector>();
        auto var = fuse_transpose(x, axes);
        forward(var);
        return;
    }
    #ifdef HAS_ACCELERATOR
    const auto backend = construction_target_backend(x);
    // cuTT (the accelerated transpose library behind OpCapability::Transpose)
    // plans its permutation kernels for fixed element sizes and does not
    // cover 16-byte elements -- `cuttPlan` fails outright for complex128
    // ("cuttPlan failed ... dsize 16"), which is a *device* error and would
    // otherwise be a hard, un-recoverable crash on the very first `.transpose`
    // of a complex128 CUDA Var (used pervasively, e.g. every `_conj_transpose`
    // call in the linalg bridge). Route those dtypes straight to the generic
    // Tx-templated kernel below instead, which has no such size assumption.
    if (backend != BackendId::Cpu && x->dtype().dsize() <= 8) {
        auto accelerated_transpose = find_op_capability<VarPtr, Var*, NanoVector>(
            backend, OpCapability::Transpose, x, axes);
        if (accelerated_transpose) {
            auto var = accelerated_transpose(x, axes);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
}

void TransposeOp::infer_shape() {
    auto xdim = x->shape.size();
    USER_CHECK(xdim);
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    } else {
        USER_CHECKop(axes.size(),==,xdim);
        int64_t mask=0;
        for (auto i : axes) mask |= 1u<<i;
        USER_CHECK(mask==((1ll<<xdim)-1)) << "Invalid axes" << axes;
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

// The permutation is read at run time, not written into the key: only the
// rank is. A tensor-network contraction meets a new permutation of a 12-D
// tensor on nearly every call, and each one used to compile a kernel of its own
// (~730 ms of g++ for a 400 us computation). The strides were always runtime
// values; the permutation only decided which stride goes with which loop, and
// that choice is now made on the host before the loops.
void TransposeOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«DIM=" << JK::hex(axes.size());
}

#else // JIT
void TransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    
    @for(i, 0, DIM, index_t yshape@i = y->shape[@i];)
    // xstride{i}: the stride in x of the axis that output axis i reads.
    @for(i, 0, DIM, index_t xstride@i = x->storage_stride(axes[@i]);)
    index_t ystride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    // `yid` is the output's linear index, so every iteration writes its own
    // element and the whole nest collapses. The depth has to be a literal: a
    // `#pragma` line is not run through the template substitution (only the
    // `@if` around it is), and building it with `_Pragma` instead makes
    // KernelIR -- which parses this file as text for the CUDA path -- read it
    // as a function definition. A transpose is a gather that does not vectorise
    // either way, so the `if` clause costs nothing here.
    @if(@is_def(JIT_cpu), index_t num = y->num;)
    @if(@is_def(JIT_cpu) && DIM==1, #pragma omp parallel for if(num >= 65536))
    @if(@is_def(JIT_cpu) && DIM==2, #pragma omp parallel for collapse(2) if(num >= 65536))
    @if(@is_def(JIT_cpu) && DIM==3, #pragma omp parallel for collapse(3) if(num >= 65536))
    @if(@is_def(JIT_cpu) && DIM>=4, #pragma omp parallel for collapse(4) if(num >= 65536))
    @for(d, 0, DIM, for (index_t yi@d=0; yi@d < yshape@d; yi@d++)) {
        auto yid = @for(d, 0, DIM, + yi@d * ystride@d);
        auto xid = @for(d, 0, DIM, + yi@d * xstride@d);
        yp[yid] = xp[xid];
    }
}
#endif // JIT

} // jittor
