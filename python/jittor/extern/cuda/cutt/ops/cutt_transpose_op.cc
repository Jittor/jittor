// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cutt_transpose_op.h"
#include "ops/op_register.h"
#include "cutt.h"
#include "cutt_warper.h"
#include "misc/stack_vector.h"
#include "helper_cuda.h"

namespace jittor {

#ifndef JIT
static auto make_transpose = get_op_info("cutt_transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();

CuttTransposeOp::CuttTransposeOp(Var* x, NanoVector axes) : x(x), axes(axes) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    int i=0;
    for (; i<axes.size(); i++)
        if (i!=axes[i]) break;
    if (i==axes.size() && axes.size()) {
        forward(x);
        return;
    }
    y = create_output(nullptr, x->dtype());
}

void CuttTransposeOp::infer_shape() {
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

VarPtr CuttTransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return make_transpose(dout, reverse);
}


void CuttTransposeOp::jit_prepare(JK& jk) {
    // do nothing
    jk << _CS("[T:1]");
}

unordered_map<string, unsigned int> cutt_plan_cache;

#else // JIT

extern unordered_map<string, unsigned int> cutt_plan_cache;

void CuttTransposeOp::jit_run() {
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
    if (dim == 1) {
        checkCudaErrors(cudaMemcpyAsync(yp, xp, x->size, cudaMemcpyDefault, 0));
        return;
    }
    jk.clear();
    jk << dim << ',';
    for (int i=0; i<dim; i++) jk << x_shape[i] << ',';
    for (int i=0; i<dim; i++) jk << reverse[i] << ',';
    jk << x->dtype().dsize() << '.';
    auto iter = cutt_plan_cache.find(jk.to_string());
    LOGvvv << "Run cutt_transpose with key:" << jk.to_string();

    if (iter!=cutt_plan_cache.end()){
        cuttExecute(iter->second, xp, yp);
    } else {
        cuttHandle plan;
        CHECK(0==cuttPlan(&plan, dim, x_shape.data(), reverse.data(), x->dtype().dsize(), 0));
        cutt_plan_cache[jk.to_string()] = plan;
        cuttExecute(plan, xp, yp);
    }
}
#endif // JIT

} // jittor