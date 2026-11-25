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

// On CPU with strided/broadcast operands, emit the ND loop directly so
// operand indices are linear affine sums of loop variables (no integer
// divisions or modulos), enabling SIMD auto-vectorization. On accelerators,
// keep the 1D loop with precomputed `zabove` divisors (omitting `/ 1` on the
// innermost axis) so MergeLoopVarPass can collapse dimensions for vectorized
// device loads. Keep `@if` bodies free of bare commas.
void BinaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ zp = z->ptr<Tz>();
    @if(@is_def(JIT_cpu) && (XSTRIDED || YSTRIDED) && DIM > 0,
        @for(d, 0, DIM, index_t zshape@d = z->shape[@d];)
        index_t zstride@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, auto zstride@d = zstride@{d+1} * zshape@{d+1};)
        @if(XSTRIDED, @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);))
        @if(YSTRIDED, @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);))
        @for(d, 0, DIM, for (index_t i@d=0; i@d < zshape@d; i@d++)) {
            index_t i = @for(d, 0, DIM, + i@d * zstride@d);
            @if(XSTRIDED,
                index_t xid = 0 @for(d, 0, DIM, @if(XSMASK>>d&1, + i@d * xstride@d));
            ,
                index_t xid = i;
            )
            @if(YSTRIDED,
                index_t yid = 0 @for(d, 0, DIM, @if(YSMASK>>d&1, + i@d * ystride@d));
            ,
                index_t yid = i;
            )
            zp[i] = @expand_op(@OP, @Tz, xp[xid], @Tx, yp[yid], @Ty);
        }
    ,
        index_t num = z->num;
        @if(XSTRIDED || YSTRIDED,
            @for(d, 1, DIM, index_t zstorage_shape@d = z->shape[@d];)
            @if(DIM > 1,
                index_t zabove@{DIM-1} = 1;
                @for(d, DIM-2, -1, -1, index_t zabove@d = zabove@{d+1} * zstorage_shape@{d+1};)
            )
        )
        @if(XSTRIDED, @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);))
        @if(YSTRIDED, @for(d, 0, DIM, index_t ystride@d = y->storage_stride(@d);))
        for (index_t i=0; i<num; i++) {
            @if(XSTRIDED,
                index_t xi = 0;
                @for(d, 0, DIM,
                    @if(XSMASK>>d&1,
                        xi += @if(d==DIM-1, @if(d, (i % zstorage_shape@d), i), @if(d, (i / zabove@d % zstorage_shape@d), (i / zabove@d))) * xstride@d;
                    )
                )
            ,
                index_t xi = i;
            )
            @if(YSTRIDED,
                index_t yi = 0;
                @for(d, 0, DIM,
                    @if(YSMASK>>d&1,
                        yi += @if(d==DIM-1, @if(d, (i % zstorage_shape@d), i), @if(d, (i / zabove@d % zstorage_shape@d), (i / zabove@d))) * ystride@d;
                    )
                )
            ,
                index_t yi = i;
            )
            zp[i] = @expand_op(@OP, @Tz, xp[xi], @Tx, yp[yi], @Ty);
        }
    )
}


} // jittor
