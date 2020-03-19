// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
    for (size_t i=0; i<xdim; i++) {
        auto xshape = x->shape[i];
        auto yshape = y->shape[i];
        auto cshape = cond->shape[i];
        CHECK(xshape==yshape && cshape==yshape) << "Shape not match";
    }
    z->set_shape(x->shape);
}

void TernaryOp::jit_prepare() {
    add_jit_define("Tc", cond->dtype());
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("Tz", z->dtype());
}

#else // JIT
void TernaryOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Tc>();
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    for (index_t i=0; i<num; i++)
        zp[i] = condp[i] ? xp[i] : yp[i];
}
#endif // JIT

} // jittor