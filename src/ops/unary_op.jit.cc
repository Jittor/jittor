// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "type/cpu_math.h"
#include "core/var.h"
#include "ops/unary_op.h"
#include "ops/layout_propagation.h"
#include "ops/op_register.h"
#include "ops/composite/array_op.h"
#include "runtime/dispatch_context.h"

namespace jittor {

// On CPU with a strided input, emit the ND loop directly so the input index
// is an affine sum of loop variables (no divisions/modulos), enabling SIMD
// auto-vectorization. Keep `@if` bodies free of bare commas.
void UnaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    @if(@is_def(JIT_cpu) && XSTRIDED && DIM > 0,
        @for(d, 0, DIM, index_t yshape@d = y->shape[@d];)
        index_t ystride@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, auto ystride@d = ystride@{d+1} * yshape@{d+1};)
        @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
        @for(d, 0, DIM, for (index_t i@d=0; i@d < yshape@d; i@d++)) {
            index_t yid = @for(d, 0, DIM, + i@d * ystride@d);
            index_t xid = 0 @for(d, 0, DIM, @if(XSMASK>>d&1, + i@d * xstride@d));
            yp[yid] = @expand_op(@OP, @Ty, xp[xid], @Tx);
        }
    ,
        index_t num = y->num;
        @if(XSTRIDED,
            @for(d, 1, DIM, index_t xshape@d = x->shape[@d];)
            @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
            @if(DIM > 1,
                index_t xabove@{DIM-1} = 1;
                @for(d, DIM-2, -1, -1, index_t xabove@d = xabove@{d+1} * xshape@{d+1};)
            )
        )
        for (index_t i=0; i<num; i++) {
            @if(XSTRIDED,
                index_t xi = 0;
                @for(d, 0, DIM,
                    @if(XSMASK>>d&1,
                        xi += @if(d==DIM-1, @if(d, (i % xshape@d), i), @if(d, (i / xabove@d % xshape@d), (i / xabove@d))) * xstride@d;
                    )
                )
            ,
                index_t xi = i;
            )
            yp[i] = @expand_op(@OP, @Ty, xp[xi], @Tx);
        }
    )
}


} // jittor
