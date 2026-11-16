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

void UnaryOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();
    index_t num = y->num;
    // The product of the shapes above each axis, so one axis' index is a single
    // division away instead of a chain of them (`(i/a)/b == i/(a*b)` for
    // non-negative integers). See KI-CODEGEN-001; keep the macro arguments
    // comma-free, the template parser splits them on commas.
    @if(XSTRIDED,
        @for(d, 0, DIM, index_t xshape@d = x->shape[@d];)
        @for(d, 0, DIM, index_t xstride@d = x->storage_stride(@d);)
        index_t xabove@{DIM-1} = 1;
        @for(d, DIM-2, -1, -1, index_t xabove@d = xabove@{d+1} * xshape@{d+1};)
    )
    for (index_t i=0; i<num; i++) {
        // XSMASK says which axes move the physical index at all, so an axis
        // outside the mask contributes no term; axis 0 needs no modulo, because
        // `i` is already below `xshape0 * xabove0` by the time it is read.
        @if(XSTRIDED,
            index_t xi = 0;
            @for(d, 0, DIM,
                @if(XSMASK>>d&1,
                    xi += @if(d, (i / xabove@d % xshape@d), (i / xabove@d)) * xstride@d;
                )
            )
        ,
            index_t xi = i;
        )
        yp[i] = @expand_op(@OP, @Ty, xp[xi], @Tx);
    }
}


} // jittor
