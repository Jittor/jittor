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

DEFINE_FLAG(int, transpose_storage_view, 0,
    "Return a permutation as a view of its input's allocation instead of a "
    "materialised copy: same storage, swapped strides, and a write through the "
    "result reaches its base -- what torch returns. Off by default: the result "
    "is then non-dense, which every consumer handles (elementwise, reduce, "
    "broadcast and reindex read storage_strides, and a slice already reaches "
    ".numpy(), .cpu() and the fused kernels this way) but a cuBLAS or cuTT path "
    "wanting dense memory may prefer one copy up front. Benchmark before "
    "making it the default.");

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
    if (backend != BackendId::Cpu) {
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
    // Decided here rather than in infer_shape so the flag is read once, at
    // construction, and the op's identity does not change under it afterwards.
    storage_view = transpose_storage_view != 0;
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
        for (auto i : axes) mask |= 1<<i;
        USER_CHECK(mask==((1ll<<xdim)-1)) << "Invalid axes" << axes;
    }
    NanoVector shape;
    for (uint i=0; i<xdim; i++)
        shape.push_back(x->shape[axes[i]]);
    y->set_shape(shape);
    if (storage_view) {
        // The permutation applied to the *input's* strides, so a transpose of
        // something already strided composes instead of assuming dense input.
        vector<int64> strides(xdim);
        for (uint i=0; i<xdim; i++)
            strides[i] = x->storage_stride(axes[i]);
        if (NanoVector::fits(strides)) {
            y->set_storage_strides(strides);
            y->share_with(x);
        } else {
            // A stride this cannot encode is not one to guess at; fall back to
            // the copy, which is always correct.
            storage_view = false;
        }
    }
}

VarPtr TransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return make_transpose(dout, reverse);
}

void TransposeOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«DIM=" << JK::hex1(axes.size());
    for (uint i=0; i<axes.size(); i++)
        jk << "«AXES" << JK::hex1(axes[i]) << '=' << JK::hex1(i);
}

#else // JIT
void TransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    
    @for(i, 0, DIM, index_t yshape@i = y->shape[@i];)
    @for(i, 0, DIM, index_t xshape@i = yshape@{AXES@i};)
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
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
        @for(d, 0, DIM, auto xi@d = yi@{AXES@d};)
        auto xid = @for(d, 0, DIM, + xi@d * xstride@d);
        yp[yid] = xp[xid];
    }
    // unused var
    (void)xshape0;
}
#endif // JIT

} // jittor
