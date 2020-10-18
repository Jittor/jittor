// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <algorithm>
#include "var.h"
#include "ops/broadcast_to_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_reduce = get_op_info("reduce")
    .get_constructor<VarPtr, Var*, NanoString, uint, uint>();
static auto make_broadcast = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, Var*, uint, uint>();
static auto make_broadcast2 = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, NanoVector, uint, uint>();
    
BroadcastToOp::BroadcastToOp(Var* x, Var* y, NanoVector dims) : x(x), y(y) {
    auto count = dims.size();
    // forward x if don't need broadcast
    if (y->num>=0 && !count && !need_broadcast(x, y->shape)) {
        forward(x);
        return;
    }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    z = create_output(NanoVector(), x->dtype());
    bcast_mask = 0;
    keepdims_mask = 0;
    auto ydim = std::max(x->shape.size(), y->shape.size()-count)+count;
    for (auto dim : dims) {
        if (dim<0) dim += ydim;
        CHECK(dim>=0 && dim<ydim) << "Wrong dims number:" << dims;
        bcast_mask |= 1 << dim; 
    }
}

BroadcastToOp::BroadcastToOp(Var* x, Var* y, uint dims_mask, uint keepdims_mask) : x(x), y(y) {
    auto count = __builtin_popcount(dims_mask);
    // forward x if don't need broadcast
    if (y->num>=0 && !count && !need_broadcast(x, y->shape)) {
        forward(x);
        return;
    }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    z = create_output(NanoVector(), x->dtype());
    bcast_mask = dims_mask;
    this->keepdims_mask = keepdims_mask;
}

BroadcastToOp::BroadcastToOp(Var* x, NanoVector shape, uint dims_mask, uint keepdims_mask) : x(x), y(nullptr), shape(shape) {
    auto count = __builtin_popcount(dims_mask);
    if (!count) {
        forward(x);
        return;
    }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    z = create_output(NanoVector(), x->dtype());
    bcast_mask = dims_mask;
    this->keepdims_mask = keepdims_mask;
}

BroadcastToOp::BroadcastToOp(Var* x, NanoVector shape, NanoVector dims) : x(x), y(nullptr), shape(shape) {
    auto count = dims.size();
    // forward x if don't need broadcast
    if (!count && !need_broadcast(x, shape)) {
        forward(x);
        return;
    }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::broadcast);
    CHECKop(shape.size(),>,0u) << "Number of shape should greater than 0.";
    for (auto v : shape)
        CHECKop(v,>=,0u) << "Shape should greater than 0.";
    z = create_output(nullptr, x->dtype());
    bcast_mask = 0;
    keepdims_mask = 0;
    auto ydim = std::max(x->shape.size(), shape.size()-count)+count;
    for (auto dim : dims) {
        if (dim<0) dim += ydim;
        CHECK(dim>=0 && dim<ydim) << "Wrong dims number:" << dims;
        bcast_mask |= 1 << dim;
    }
}

bool BroadcastToOp::need_broadcast(const Var* x, const NanoVector& shape) {
    if (x->shape.size() < shape.size()) return true;
    for (uint i=shape.size()-1, j=x->shape.size()-1; i<shape.size(); i--,j--)
        if (x->shape[j]< 0 || (x->shape[j] != shape[i] && shape[i] != 1)) return true;
    return false;
}

VarPtr BroadcastToOp::duplicate() {
    if (y)
        return make_broadcast(x, y, bcast_mask, keepdims_mask);
    else
        return make_broadcast2(x, shape, bcast_mask, keepdims_mask);
}

VarPtr BroadcastToOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index==1) return nullptr;
    if (bcast_mask==0) return dout;
    VarPtr dv = make_reduce(dout, ns_add, bcast_mask, keepdims_mask);
    if (dv->shape.size() != v->shape.size())
        dv->shape = v->shape;
    return dv;
}

void BroadcastToOp::infer_shape() {
    if (y && y->num>=0) {
        // shape of y is already solved, we can remove deps
        LOGvvvv << "Remove broadcast y deps" << y;
        shape = y->shape;
        set_inputs({x});
        y = nullptr;
    }
    auto yshapes = y ? y->shape : shape;
    auto xdim = x->shape.size();
    auto ydim = yshapes.size();
    auto count = __builtin_popcount(bcast_mask&~keepdims_mask);
    auto zdim = std::max(xdim, ydim-count) + count;
    
    int64 zz[zdim];
    for (int i=zdim-1, xi = xdim-1, yi = ydim-1; i>=0; i--) {
        bool bx = xi>=0;
        bool by = yi>=0;
        auto xshape = bx ? x->shape[xi] : 1;
        auto yshape = by ? yshapes[yi] : 1;
        if (bcast_mask>>i&1) {
            yi--;
            if (keepdims_mask>>i&1) xi--;
            zz[i] = yshape;
            continue;
        }
        auto mask = ((xshape==1 && (yshape!=1 || !bx))&1) << i;
        bcast_mask |= mask;
        keepdims_mask |= mask;
        int64 zs;
        if ((xshape == 1 || yshape == 1) && (xshape != yshape)) {
            zs = xshape * yshape;
        } else if (xshape < 0 || yshape < 0) {
            zs = std::min(xshape, yshape);
        } else {
            CHECKop(xshape,==,yshape) << "Shape not match" << x->shape << yshapes << bcast_mask;
            zs = xshape;
        }
        zz[i] = zs;
        xi--, yi--;
    }

    NanoVector zshape;
    for (int i=0; i<zdim; i++) zshape.push_back(zz[i]);
    z->set_shape(zshape);
    LOGvvv << "Broadcast x(" >> x >> ") shape" << yshapes << "-> z(" >> z >> ")"; 
}

void BroadcastToOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("DIM", JK::hex1(z->shape.size()));
    add_jit_define("BCAST", JK::hex(bcast_mask));
}

#else // JIT
void BroadcastToOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ zp = z->ptr<Tx>();
    // define z shape
    @for(i, 0, DIM, index_t zshape@i = z->shape[@i];)
    // define z stride
    index_t zstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto zstride@i = zstride@{i+1} * zshape@{i+1};)
    // define x stride
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * @if(BCAST>>(i+1)&1,1,zshape@{i+1});)
    // generate d-for loop
    @for(d, 0, DIM, for (index_t i@d=0; i@d < zshape@d; i@d++)) {
        auto zid = @for(d, 0, DIM, + i@d * zstride@d);
        auto xid = @for(d, 0, DIM, + @if(BCAST>>d&1,0,i@d) * xstride@d);
        zp[zid] = xp[xid];
    }
}
#endif // JIT

} // jittor