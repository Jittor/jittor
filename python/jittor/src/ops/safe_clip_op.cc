// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/safe_clip_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

SafeClipOp::SafeClipOp(Var* x, float64 left, float64 right) : x(x), left(left), right(right) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    set_type(OpType::element);
    y = create_output(nullptr, x->dtype());
}

VarPtr SafeClipOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return dout;
}

void SafeClipOp::infer_shape() {
    y->set_shape(x->shape);
}

void SafeClipOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype() <<']';
}

#else // JIT
void SafeClipOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    Tx left_value = (Tx)std::max((float64)std::numeric_limits<Tx>::lowest(), left);
    Tx right_value = (Tx)std::min((float64)std::numeric_limits<Tx>::max(), right);
    auto* __restrict__ yp = y->ptr<Tx>();
    index_t num = y->num;
    for (index_t i=0; i<num; i++)
        yp[i] = xp[i] < left_value ? left_value : (xp[i] > right_value ? right_value : xp[i]);
}
#endif // JIT

} // jittor