// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "core/var.h"
#include "ops/binary_op.h"
#include "ops/layout_propagation.h"
#include "ops/broadcast_to_op.h"
#include "ops/op_register.h"
#include "ops/composite/array_op.h"

namespace jittor {

void BinaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    index_t num = z->num;
    // The product of the shapes *above* each axis, so one axis' index is a
    // single division away (`i / zabove@d % zstorage_shape@d`) instead of a
    // chain of them. `(i/a)/b == i/(a*b)` for non-negative integers so the
    // chain the recovery used to emit -- two divisions plus a modulo per element
    // for the common one-strided-axis case and all of them serialised --
    // collapses to one division and one modulo per masked axis. See
    // KI-CODEGEN-001. (Keep the macro arguments below comma-free: the template
    // parser splits them on commas.)
    @if(XSTRIDED || YSTRIDED,
        @for(d, 0, DIM, index_t zstorage_shape@d = z->shape[@d];)
        index_t zabove@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, index_t zabove@d = zabove@{d+1} * zstorage_shape@{d+1};)
    )
    @if(XSTRIDED, @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);))
    @if(YSTRIDED, @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);))
    for (index_t i=0; i<num; i++) {
        // X/YSMASK say which axes move the physical index at all, so an axis
        // outside the mask contributes no term; axis 0 needs no modulo, because
        // `i` is already below `zstorage_shape0 * zabove0` by the time it is
        // read.
        @if(XSTRIDED,
            index_t xi = 0;
            @for(d, 0, DIM,
                @if(XSMASK>>d&1,
                    xi += @if(d, (i / zabove@d % zstorage_shape@d), (i / zabove@d)) * xstride@d;
                )
            )
        ,
            index_t xi = i;
        )
        @if(YSTRIDED,
            index_t yi = 0;
            @for(d, 0, DIM,
                @if(YSMASK>>d&1,
                    yi += @if(d, (i / zabove@d % zstorage_shape@d), (i / zabove@d)) * ystride@d;
                )
            )
        ,
            index_t yi = i;
        )
        zp[i] = @expand_op(@OP, @Tz, xp[xi], @Tx, yp[yi], @Ty);
    }
}


} // jittor
