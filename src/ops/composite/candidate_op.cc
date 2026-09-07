// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/composite/candidate_op.h"

namespace jittor {

#ifndef JIT
CandidateOp::CandidateOp(Var* x, string&& fail_cond, NanoString dtype) : x(x), fail_cond(move(fail_cond)) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, dtype);
}

void CandidateOp::infer_shape() {
    y->set_shape({-x->shape[0]});
}

void CandidateOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«FUNC:" << fail_cond;
    jk << "«XDIM=" << JK::hex1(x->shape.size());
}

#else // JIT
void CandidateOp::jit_run() {
    using namespace std;
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
