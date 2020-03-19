// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/candidate_op.h"

namespace jittor {

#ifndef JIT
CandidateOp::CandidateOp(Var* x, string&& fail_cond, NanoString dtype) : x(x), fail_cond(move(fail_cond)) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_vary_shape);
    y = create_output(nullptr, dtype);
}

void CandidateOp::infer_shape() {
    y->set_shape({-std::abs(x->shape[0])});
}

void CandidateOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("FUNC", fail_cond);
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
}

#else // JIT
void CandidateOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    // define cond shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define cond stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)
    
    // define ys
    auto* __restrict__ yp = y->ptr<Ty>();
    int64 n=0;

    // generate d-for loop
    for (index_t i=0; i < xshape0; i++) {
        bool pass = true;
        for (index_t j_=0; j_ < n; j_++) {
            index_t j = yp[j_];
            if (@FUNC) {
                pass = false;
                break;
            }
        }
        if (pass) {
            yp[n] = i;
            n++;
        }
    }
    y->set_shape({n});
}
#endif // JIT

} // jittor