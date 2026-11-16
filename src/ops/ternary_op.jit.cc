// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/ternary_op.h"
#include "ops/layout_propagation.h"
#include "ops/op_register.h"

namespace jittor {

void TernaryOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Tc>();
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    // The product of the shapes above each axis, so one axis' index is a single
    // division away (`i / zabove@d % zstorage_shape@d`) instead of a chain of
    // them; `(i/a)/b == i/(a*b)` for non-negative integers. See KI-CODEGEN-001,
    // and keep the macro arguments comma-free -- the template parser splits them
    // on commas.
    @if(TSMASK,
        @for(d, 0, DIM, index_t zstorage_shape@d = z->shape[@d];)
        index_t zabove@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, index_t zabove@d = zabove@{d+1} * zstorage_shape@{d+1};)
    )
    @if(XSTRIDED, @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);))
    @if(YSTRIDED, @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);))
    @if(CSTRIDED, @for(d, 0, DIM, index_t cstride@d = cond->storage_stride(@d);))
    for (index_t i=0; i<num; i++) {
        // One axis index shared by whichever operands are strided on it; axis 0
        // needs no modulo because `i` is already below its extent.
        @if(XSTRIDED, index_t xi=0;, index_t xi=i;)
        @if(YSTRIDED, index_t yi=0;, index_t yi=i;)
        @if(CSTRIDED, index_t ci=0;, index_t ci=i;)
        @for(d, 0, DIM,
            @if(TSMASK>>d&1,
                index_t q@d = @if(d, (i / zabove@d % zstorage_shape@d), (i / zabove@d));
                @if(XSMASK>>d&1, xi += q@d * xstride@d;)
                @if(YSMASK>>d&1, yi += q@d * ystride@d;)
                @if(CSMASK>>d&1, ci += q@d * cstride@d;)
            )
        )
        Tz xd_ = xp[xi];
        Tz yd_ = yp[yi];
        zp[i] = condp[ci] ? xd_ : yd_;
    }
}


} // jittor
