// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include <algorithm>
#include "core/var.h"
#include "ops/broadcast_to_op.h"
#include "ops/op_register.h"

namespace jittor {

void BroadcastToOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ zp = z->ptr<Tx>();
    // define z shape
    @for(i, 0, DIM, index_t zshape@i = z->shape[@i];)
    // define z stride
    index_t zstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto zstride@i = zstride@{i+1} * zshape@{i+1};)
    // define x stride
    index_t xstride@{DIM-1} = 1;
    @for(i, DIM-2, -1, -1, auto xstride@i = xstride@{i+1} * @if(BCAST>>(i+1)&1,1,zshape@{i+1});)
    // generate d-for loop
    @for(d, 0, DIM, for (index_t i@d=0; i@d < zshape@d; i@d++)) {
        auto zid = @for(d, 0, DIM, + i@d * zstride@d);
        auto xid = @for(d, 0, DIM, + @if(BCAST>>d&1,0,i@d) * xstride@d);
        zp[zid] = xp[xid];
    }
}


} // jittor
