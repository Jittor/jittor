// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/where_op.h"

namespace jittor {

#ifndef JIT
WhereOp::WhereOp(Var* cond, NanoString dtype) : cond(cond) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_vary_shape);
    auto ndim = cond->shape.size();
    outs.reset(new Var*[ndim]);
    for (uint i=0; i<ndim; i++)
        outs[i] = create_output(nullptr, dtype);
}

void WhereOp::infer_shape() {
    auto ndim = cond->shape.size();
    auto num = cond->num;
    if (num>0) num = -num;
    for (uint i=0; i<ndim; i++)
        outs[i]->set_shape({num});
}

void WhereOp::jit_prepare() {
    add_jit_define("Ti", cond->dtype());
    add_jit_define("To", outs[0]->dtype());
    add_jit_define("NDIM", JK::hex1(cond->shape.size()));
}

#else // JIT
void WhereOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Ti>();
    // define cond shape
    @for(i, 0, NDIM, index_t condshape@i = cond->shape[@i];)
    // define cond stride
    index_t condstride@{NDIM-1} = 1;
    @for(i, NDIM-2, -1, -1, auto condstride@i = condstride@{i+1} * condshape@{i+1};)
    
    // define outs
    @for(i, 0, NDIM,  auto* __restrict__ outs@i@@p = outs[@i]->ptr<To>();)
    int64 n=0;

    // generate d-for loop
    @for(d, 0, NDIM, for (index_t i@d=0; i@d < condshape@d; i@d++)) {
        auto condid = @for(d, 0, NDIM, + i@d * condstride@d);
        if (condp[condid]) {
            @for(i, 0, NDIM, outs@i@@p[n] = i@i;)
            n++;
        }
    }
    @for(i, 0, NDIM, outs[@i]->set_shape({n});)
}
#endif // JIT

} // jittor