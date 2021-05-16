// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <mkldnn.hpp>

#include "var.h"
#include "mkl_matmul_op.h"

using namespace mkldnn;
using namespace std;

namespace jittor {

#ifndef JIT

MklMatmulOp::MklMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == 4 && b->dtype().dsize() == 4) << "support float32 only now.";
    c = create_output(nullptr, a->dtype());
}

void MklMatmulOp::infer_shape() {
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

void MklMatmulOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << a->dtype();
    jk << _CS("][Trans_a:") << (trans_a ? 'T' : 'N');
    jk << _CS("][Trans_b:") << (trans_b ? 'T' : 'N') << ']';
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
void MklMatmulOp::jit_run() {
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
    ASSERTop(0,==,mkldnn_sgemm('@Trans_a', '@Trans_b', n, k, m,
        1.f, a->ptr<T>(), '@Trans_a'=='N'? m : n,
        b->ptr<T>(), '@Trans_b' == 'N' ? k : m,
        0.f, c->ptr<T>(), k));
}
#endif
#endif // JIT

} // jittor
