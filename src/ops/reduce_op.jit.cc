// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <limits>
#include "core/var.h"
#include "ops/reduce_op.h"
#include "ops/op_register.h"
#include "core/executor.h"
#include "runtime/backend.h"
#include "runtime/device.h"

namespace jittor {

void ReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Ty>();

    @for(i, 0, DIM, index_t xshape@i = x->shape[@i];)
    @for(i, 0, DIM, index_t yshape@i = @if(REDUCE>>i&1,1,xshape@i);)
    // A rank-0 input reduces over no dimensions at all, and `@{DIM-1}` is
    // then `-1`: this line used to emit `index_t ystride-1 = 1;`, which does
    // not compile. `loss.sum()` where the loss is already a scalar is ordinary
    // code -- PyTorch returns the value unchanged -- and it died here on both
    // devices. With the guard, every `@for` below produces an
    // empty nest, the body runs once with `yid == xid == 0`, and the result is
    // the input value, which is what the reduction of a single element is.
    @if(DIM>0, index_t ystride@{DIM-1} = 1;)
    @for(i, DIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)
    // A contiguous input's storage strides *are* the row-major chain of its own
    // shape -- `Var::storage_stride` returns exactly this product when the var
    // carries no explicit stride vector, and `is_contiguous` is true precisely
    // when it does not (a size-1 axis may carry any stride; it is never read,
    // because its loop bound is 1). Deriving the strides from `xshape` keeps
    // `xid` a linear combination of the loop ids, so the tracer folds it and
    // `MergeLoopVarPass` can merge the two outer loops. Reading
    // `x->storage_stride` instead makes every stride an opaque call, `xid` stops
    // matching `id_a*range_b*d + id_b*d + c`, and the merge is refused: two
    // parallel dimensions instead of one, and no constant folding at all. That
    // is what happened to this op in 7e83d6da and is recorded under
    // KI-CODEGEN-001. A strided view (a broadcast, a stepped slice) has no
    // shape-derived strides, so it keeps the run-time read.
    @if(XSTRIDED,
        @for(i, 0, DIM, index_t xstride@i = x->storage_stride(@i);)
    ,
        @if(DIM>0, index_t xstride@{DIM-1} = 1;)
        @for(i, DIM-2, -1, -1, index_t xstride@i = xstride@{i+1} * xshape@{i+1};)
    )
    Ty count = x->num*1.0 / y->num;
    Ty rcount = y->num*1.0 / x->num;
    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
        yp[yid] = @expand_op(init_@OP, @Ty);
        @if(@EMPTY_MEAN,
            yp[yid] = (Ty)::nanf("");
            ,
        )
    }

    @for(d, 0, DIM,@if(REDUCE>>d&1,, for (index_t xi@d=0; xi@d < xshape@d; xi@d++))) {
        @for(d, 0, DIM,@if(REDUCE>>d&1, for (index_t xi@d=0; xi@d < xshape@d; xi@d++),)) {
            auto yid = 0 @for(d, 0, DIM,@if(REDUCE>>d&1,, + xi@d * ystride@d));
            auto xid = 0 @for(d, 0, DIM, + xi@d * xstride@d);
            yp[yid] = @expand_op(@OP, @Ty, yp[yid], @Ty, xp[xid], @Tx);
        }
    }
    (void)count; (void)rcount;
    @if(DIM>0, (void)yshape0; (void)ystride0;)
}


} // jittor
