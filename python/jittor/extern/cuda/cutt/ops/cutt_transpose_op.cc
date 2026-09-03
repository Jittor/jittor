// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cutt_transpose_op.h"
#include "ops/op_register.h"
#include "cutt.h"
#include "cutt_wrapper.h"
#include "misc/stack_vector.h"
#include "helper_cuda.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = op_constructor<VarPtr, Var*, NanoVector>("cutt_transpose");

CuttTransposeOp::CuttTransposeOp(Var* x, NanoVector axes) : x(x), axes(axes) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    int i=0;
    for (; i<axes.size(); i++)
        if (i!=axes[i]) break;
    if (i==axes.size() && axes.size()) {
        forward(x);
        return;
    }
    y = create_output(nullptr, x->dtype());
    set_flag(OpFlags::_manual_set_vnbb);
}

void CuttTransposeOp::infer_shape() {
    auto xdim = x->shape.size();
    CHECK(xdim);
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

VarPtr CuttTransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return make_transpose(dout, reverse);
}


void CuttTransposeOp::jit_prepare(JK& jk) {
    // do nothing
    jk << "«T:1";
}

#else // JIT

void CuttTransposeOp::jit_run() {
    // Return if x is empty
    if (x->num == 0)
        return;

    auto* __restrict__ xp = x->mem_ptr;
    auto* __restrict__ yp = y->mem_ptr;
    StackVector<int> x_shape;
    StackVector<int> new_shape, new_axes, trans, reverse;
    int dim = x->shape.size();
    for (int i=0; i<dim; i++) {
        trans[i] = new_shape.size();
        if (x->shape[i] != 1)
            new_shape.push_back(x->shape[i]);
    }
    for (int i = 0; i < dim; ++i) {
        if (x->shape[axes[i]] != 1) {
            new_axes.push_back(trans[axes[i]]);
        }
    }
    dim = new_shape.size();
    for (int i=0; i<dim; i++)
        reverse[i] = dim-1-new_axes[dim-1-i];
    for (int i=0; i<dim; i++)
        x_shape[i] = new_shape[dim-1-i];
    if (dim == 1 || x->num==1) {
        checkCudaErrors(cudaMemcpyAsync(yp, xp, x->size, cudaMemcpyDeviceToDevice, 0));
        return;
    }
    // The plan key is a POD whose bytes are compared directly: no string is
    // built here, and the shared global JIT key buffer (which also serves the
    // executor) is left alone.
    ASSERT(dim <= CUTT_PLAN_MAX_RANK)
        << "cutt_transpose supports at most" << CUTT_PLAN_MAX_RANK
        << "non-unit dimensions, got" << dim;
    CuttPlanKey key;
    std::memset(&key, 0, sizeof(key));
    key.rank = dim;
    key.dsize = x->dtype().dsize();
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    key.device = device;
    for (int i=0; i<dim; i++) {
        key.shape[i] = x_shape[i];
        key.permutation[i] = reverse[i];
    }
    LOGvvv << "Run cutt_transpose with key rank=" >> dim >> " dsize=" >> key.dsize;
    cuttExecute(cutt_get_plan(key), xp, yp);
}
#endif // JIT

} // jittor
