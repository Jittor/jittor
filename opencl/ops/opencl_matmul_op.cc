// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "opencl_matmul_helper.h"
#include "opencl_matmul_op.h"
#include "common.h"
using namespace std;

namespace jittor {

#ifndef JIT

OpenclMatmulOp::OpenclMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == b->dtype().dsize()) << "type of two inputs should be the same";
    c = create_output(nullptr, a->dtype());
}

void OpenclMatmulOp::infer_shape() {
    ASSERTop(a->shape.size(),==,2);
    ASSERTop(b->shape.size(),==,2);
    int n = a->shape[0], m = a->shape[1];
    int m_ = b->shape[0], k = b->shape[1];
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    ASSERTop(m,==,m_);
    c->set_shape({n, k});
}

void OpenclMatmulOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << a->dtype();
    jk << _CS("][Trans_a:") << (trans_a ? 'T' : 'N');
    jk << _CS("][Trans_b:") << (trans_b ? 'T' : 'N');
    jk << _CS("][op:") << (a->dtype().dsize() == 4 ? 'S' : 'D');
    jk << ']';
}

#else // JIT
#pragma clang diagnostic ignored "-Wtautological-compare"
void OpenclMatmulOp::jit_run() {
    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto n = as[0];
    auto m = as[1];
    auto k = bs[1];
    if ('@Trans_a'=='T') {
        n = as[1];
        m = as[0];
    }
    if ('@Trans_b'=='T') {
        k = bs[0];
    }
    // a: [n,m], b: [m,k], c: [n,k]
    myclblas(a->ptr<T>(),b->ptr<T>(),c->ptr<T>(),k,m,n);
}
#endif // JIT

} // jittor
