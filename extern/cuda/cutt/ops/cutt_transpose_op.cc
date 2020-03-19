// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cutt_transpose_op.h"
#include "ops/op_register.h"
#include <iostream>

#ifdef JIT
#include "cutt.h"
#endif
#include "cutt_warper.h"

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

void CuttTransposeOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("DIM", JK::hex1(axes.size()));
    for (uint i=0; i<axes.size(); i++)
        add_jit_define("AXES", JK::hex1(axes[i]), S(i));
}
unordered_map<string, unsigned int> cutt_plan_cache;

#else // JIT
#ifdef JIT_cuda

extern unordered_map<string, unsigned int> cutt_plan_cache;

void CuttTransposeOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    vector<int> permutation, permutation2;
    vector<int> y_shape;
    vector<int> x_shape;
    @for(i, 0, DIM, permutation.push_back(DIM-1-AXES@i);)
    @for(i, 0, DIM, permutation2.push_back(permutation[DIM-1-@i@@]);)
    std::vector<int> reverse;
    reverse.reserve(permutation2.size());
    for (uint i=0; i<permutation2.size(); i++)
        reverse[permutation2[i]] = i;

    @for(i, 0, DIM, x_shape.push_back(x->shape[DIM-1-@i@@]);)

    jk.clear();
    jk << @DIM << ",";
    for (uint i=0; i<@DIM; i++) jk << x_shape[i] << ",";
    for (uint i=0; i<@DIM; i++) jk << reverse[i] << ",";
    jk << sizeof(Tx) << ".";
    auto iter = cutt_plan_cache.find(jk.to_string());

    if (iter!=cutt_plan_cache.end()){
        cuttExecute(iter->second, xp, yp);
    } else {
        cuttHandle plan;
        cuttPlan(&plan, @DIM, x_shape.data(), reverse.data(), sizeof(Tx), 0);
        cutt_plan_cache[jk.to_string()] = plan;
        cuttExecute(plan, xp, yp);
    }
}
#endif // JIT_cuda
#endif // JIT

} // jittor