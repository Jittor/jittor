// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/ternary_op.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
static auto make_broadcast = op_constructor<VarPtr, Var*, Var*, NanoVector>("broadcast_to");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");

TernaryOp::TernaryOp(Var* cond, Var* x, Var* y) : cond(cond), x(x), y(y) {
    bool bx = cond->shape.size() > x->shape.size() || cond->num > x->num;
    bool by = cond->shape.size() > y->shape.size() || cond->num > y->num;
    bool bx2 = cond->shape.size() < x->shape.size() || cond->num < x->num;
    bool by2 = cond->shape.size() < y->shape.size() || cond->num < y->num;
    if (bx || by || bx2 || by2) {
        VarPtr xx, yy, cc;
        if (bx2) cc = make_broadcast(cond, x, NanoVector()), cond=cc;
        if (by2) cc = make_broadcast(cond, y, NanoVector()), cond=cc;
        bx = cond->shape.size() > x->shape.size() || cond->num > x->num;
        by = cond->shape.size() > y->shape.size() || cond->num > y->num;
        if (bx) xx = make_broadcast(x, cond, NanoVector()), x = xx;
        if (by) yy = make_broadcast(y, cond, NanoVector()), y = yy;
        forward(make_ternary(cond, x, y));
        return;
    }
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_type(OpType::element);
    set_flag(OpFlags::_manual_set_vnbb);
    cond->set_flag(VarFlags::_needed_by_backward);
    if (x->dtype() == y->dtype()) {
        z = create_output(nullptr, x->dtype());
    } else {
        z = create_output(nullptr, dtype_infer(x->ns, y->ns, x->flag(VarFlags::_is_scalar), y->flag(VarFlags::_is_scalar)));
    }
}

VarPtr TernaryOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index==0) return nullptr;
    VarPtr grad = dout;
    if (grad->dtype() != v->dtype())
        grad = make_unary(grad, v->dtype());
    auto zeros = make_number(0, v);
    if (v_index==1)
        return make_ternary(cond, grad, zeros);
    else
        return make_ternary(cond, zeros, grad);
}

void TernaryOp::infer_shape() {
    auto xdim = x->shape.size();
    auto ydim = y->shape.size();
    auto cdim = cond->shape.size();
    USER_CHECK(xdim==ydim && cdim==ydim) << "Number of dims should be the same.";
    NanoVector zshape;
    for (size_t i=0; i<xdim; i++) {
        auto xshape = x->shape[i];
        auto yshape = y->shape[i];
        auto cshape = cond->shape[i];
        auto shape = std::min(xshape, std::min(yshape, cshape));
        auto shape2 = std::max(xshape, std::max(yshape, cshape));
        zshape.push_back(shape2);
        USER_CHECK(shape==shape2) << "Shape not match" << x->shape << y->shape << cond->shape;
    }
    z->set_shape(zshape);
}

void TernaryOp::jit_prepare(JK& jk) {
    jk << "«Tc:" << cond->dtype();
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tz:" << z->dtype();
    jk << "«DIM=" << JK::hex1(z->shape.size());
    jk << "«STRIDED=" << JK::hex1(!x->is_contiguous() || !y->is_contiguous() || !cond->is_contiguous());
}

#else // JIT
void TernaryOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Tc>();
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    @if(STRIDED,
        @for(d, 0, DIM, index_t zstorage_shape@d = z->shape[@d];)
        @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
        @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);)
        @for(d, 0, DIM, index_t cstride@d = cond->storage_stride(@d);)
    )
    for (index_t i=0; i<num; i++) {
        index_t xi=i;
        index_t yi=i;
        index_t ci=i;
        @if(STRIDED,
            index_t rem=i; xi=0; yi=0; ci=0;
            @for(d, DIM-1, -1, -1,
                xi += (rem % zstorage_shape@d) * xstride@d;
                yi += (rem % zstorage_shape@d) * ystride@d;
                ci += (rem % zstorage_shape@d) * cstride@d;
                rem /= zstorage_shape@d;
            )
        )
        Tz xd_ = xp[xi];
        Tz yd_ = yp[yi];
        zp[i] = condp[ci] ? xd_ : yd_;
    }
}
#endif // JIT

} // jittor
