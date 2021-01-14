// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/ternary_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_ternary = get_op_info("ternary")
    .get_constructor<VarPtr, Var*, Var*, Var*>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();

TernaryOp::TernaryOp(Var* cond, Var* x, Var* y) : cond(cond), x(x), y(y) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    z = create_output(nullptr, dtype_infer(x->ns, y->ns));
}

VarPtr TernaryOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index==0) return nullptr;
    auto zeros = make_number(0, dout);
    if (v_index==1)
        return make_ternary(cond, dout, zeros);
    else
        return make_ternary(cond, zeros, dout);
}

void TernaryOp::infer_shape() {
    auto xdim = x->shape.size();
    auto ydim = y->shape.size();
    auto cdim = cond->shape.size();
    CHECK(xdim==ydim && cdim==ydim) << "Number of dims should be the same.";
    NanoVector zshape;
    for (size_t i=0; i<xdim; i++) {
        auto xshape = x->shape[i];
        auto yshape = y->shape[i];
        auto cshape = cond->shape[i];
        auto shape = std::min(xshape, std::min(yshape, cshape));
        auto shape2 = std::max(xshape, std::max(yshape, cshape));
        zshape.push_back(shape2);
        if (shape < 0) continue;
        CHECK(shape==shape2) << "Shape not match" << x->shape << y->shape << cond->shape;
    }
    z->set_shape(zshape);
}

void TernaryOp::jit_prepare(JK& jk) {
    jk << _CS("[Tc:") << cond->dtype();
    jk << _CS("][Tx:") << x->dtype();
    jk << _CS("][Ty:") << y->dtype();
    jk << _CS("][Tz:") << z->dtype() << ']';
}

#else // JIT
void TernaryOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Tc>();
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    for (index_t i=0; i<num; i++) {
        Tz xd_ = xp[i];
        Tz yd_ = yp[i];
        zp[i] = condp[i] ? xd_ : yd_;
    }
}
#endif // JIT

} // jittor