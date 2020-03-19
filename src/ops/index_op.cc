// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/index_op.h"

namespace jittor {

#ifndef JIT
IndexOp::IndexOp(NanoVector shape, int64 dim, NanoString dtype) : dim(dim) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    x.reset(new Var*[1]);
    x[0] = create_output(shape, dtype);
}

IndexOp::IndexOp(NanoVector shape, NanoString dtype) : dim(shape.size()) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    x.reset(new Var*[dim]);
    for (int i=0; i<dim; i++)
        x[i] = create_output(shape, dtype);
}

IndexOp::IndexOp(Var* a, int64 dim, NanoString dtype) : dim(dim) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    x.reset(new Var*[1]);
    x[0] = create_output(nullptr, dtype);
}

IndexOp::IndexOp(Var* a, NanoString dtype) : dim(a->shape.size()) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    x.reset(new Var*[dim]);
    for (int i=0; i<dim; i++)
        x[i] = create_output(nullptr, dtype);
}

void IndexOp::infer_shape() {
    if (!inputs().size()) return;
    Var* a = inputs().front();
    for (Var* o : outputs())
        o->set_shape(a->shape);
}

void IndexOp::jit_prepare() {
    add_jit_define("T", x[0]->dtype());
    add_jit_define("DIM", JK::hex1(dim));
    add_jit_define("XDIM", JK::hex1(x[0]->shape.size()));
}

#else // JIT
void IndexOp::jit_run() {
    @if(DIM==XDIM,
        @for(i,0,XDIM, auto* __restrict__ x@i@@p = x[@i]->ptr<T>();)
    ,
        auto* __restrict__ x0p = x[0]->ptr<T>();
    )
    // define x shape
    @for(i, 0, XDIM, index_t x0shape@i = x[0]->shape[@i];)
    // define x stride
    index_t x0stride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto x0stride@i = x0stride@{i+1} * x0shape@{i+1};)
    
    @for(d, 0, XDIM, for (index_t i@d=0; i@d < x0shape@d; i@d++)) {
        auto xid = @for(d, 0, XDIM, + i@d * x0stride@d);
        @if(DIM==XDIM,
            @for(i,0,XDIM, x@i@@p[xid] = i@i;)
        ,
            x0p[xid] = i@DIM;
        )
    }
}
#endif // JIT

} // jittor