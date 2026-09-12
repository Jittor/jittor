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
    // Per operand, not one flag for all three: a strided condition used to put
    // the full unflattening on the two value operands as well, which are
    // usually the contiguous ones.
    bool cs = !cond->is_contiguous(), xs = !x->is_contiguous(), ys = !y->is_contiguous();
    int cmask = cs ? cond->stride_pattern() : 0;
    int xmask = xs ? x->stride_pattern() : 0;
    int ymask = ys ? y->stride_pattern() : 0;
    jk << "«Tc:" << cond->dtype();
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tz:" << z->dtype();
    jk << "«DIM=" << JK::hex1(z->shape.size());
    jk << "«CSTRIDED=" << JK::hex1(cs);
    jk << "«XSTRIDED=" << JK::hex1(xs);
    jk << "«YSTRIDED=" << JK::hex1(ys);
    // All four unconditionally: the template reads a per-operand mask from
    // inside the shared chain's `@if(TSMASK...)`, which a contiguous operand
    // still reaches, so leaving its mask out of the key would leave the
    // template with an undefined name rather than a zero.
    jk << "«CSMASK=" << JK::hex(cmask);
    jk << "«XSMASK=" << JK::hex(xmask);
    jk << "«YSMASK=" << JK::hex(ymask);
    // The axes any strided operand actually moves. One shared remainder chain
    // serves all three, so an axis none of them moves costs nothing.
    jk << "«TSMASK=" << JK::hex(cmask | xmask | ymask);
}

#else // JIT
void TernaryOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Tc>();
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    @if(TSMASK,
        @for(d, 0, DIM, index_t zstorage_shape@d = z->shape[@d];)
    )
    @if(XSTRIDED, @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);))
    @if(YSTRIDED, @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);))
    @if(CSTRIDED, @for(d, 0, DIM, index_t cstride@d = cond->storage_stride(@d);))
    for (index_t i=0; i<num; i++) {
        // One remainder chain shared by whichever operands are strided. An axis
        // no strided operand moves is skipped entirely; axis 0 needs no modulo
        // because `i` is already below its extent; the division at an axis
        // survives only while a lower axis still reads `rem`.
        @if(XSTRIDED, index_t xi=0;, index_t xi=i;)
        @if(YSTRIDED, index_t yi=0;, index_t yi=i;)
        @if(CSTRIDED, index_t ci=0;, index_t ci=i;)
        @if(TSMASK, index_t rem=i;)
        @for(d, DIM-1, -1, -1,
            @if(TSMASK>>d&1,
                index_t q@d = @if(d, rem % zstorage_shape@d, rem);
                @if(XSMASK>>d&1, xi += q@d * xstride@d;)
                @if(YSMASK>>d&1, yi += q@d * ystride@d;)
                @if(CSMASK>>d&1, ci += q@d * cstride@d;)
            )
            @if(TSMASK&((1<<d)-1), rem /= zstorage_shape@d;)
        )
        Tz xd_ = xp[xi];
        Tz yd_ = yp[yi];
        zp[i] = condp[ci] ? xd_ : yd_;
    }
}
#endif // JIT

} // jittor
