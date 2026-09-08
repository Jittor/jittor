// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/composite/where_op.h"
#include "runtime/device.h"
#include "ops/op_register.h"
#include "ops/composite/op_capability.h"

namespace jittor {

#ifndef JIT
WhereOp::WhereOp(Var* cond, NanoString dtype) : cond(cond) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    auto ndim = cond->shape.size();
    #ifdef HAS_ACCELERATOR
    const auto backend = construction_target_backend(cond);
    if (backend != BackendId::Cpu) {
        auto accelerated_where = find_op_capability<std::vector<VarPtr>, Var*, NanoString>(
            backend, OpCapability::Where, cond, dtype);
        if (accelerated_where) {
            auto var = accelerated_where(cond, dtype);
            for(uint i=0;i<ndim;i++)
                forward(var[i]);
            return;
        }
    }
    #endif
    outs.reset(new Var*[ndim]);
    for (uint i=0; i<ndim; i++)
        outs[i] = create_output(nullptr, dtype);
}
static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");
WhereOp::WhereOp(Var* cond, Var* x, Var* y) {
    forward(make_ternary(cond, x, y));
    return;
}

void WhereOp::infer_shape() {
    auto ndim = cond->shape.size();
    auto num = -cond->num;
    for (uint i=0; i<ndim; i++)
        outs[i]->set_shape({num});
}

void WhereOp::jit_prepare(JK& jk) {
    jk << "«Ti:" << cond->dtype();
    jk << "«To:" << outs[0]->dtype();
    jk << "«NDIM=" << JK::hex1(cond->shape.size());
}

#else // JIT
void WhereOp::jit_run() {
    auto* __restrict__ condp = cond->ptr<Ti>();
    // define cond shape
    @for(i, 0, NDIM, index_t condshape@i = cond->shape[@i];)
    // define cond stride
    index_t condstride@{NDIM-1} = 1;
    @for(i, NDIM-2, -1, -1, auto condstride@i = condstride@{i+1} * condshape@{i+1};)
    
    // define outs
    @for(i, 0, NDIM,  auto* __restrict__ outs@i@@p = outs[@i]->ptr<To>();)
    int64 n=0;

    // generate d-for loop
    @for(d, 0, NDIM, for (index_t i@d=0; i@d < condshape@d; i@d++)) {
        auto condid = @for(d, 0, NDIM, + i@d * condstride@d);
        if (condp[condid]) {
            @for(i, 0, NDIM, outs@i@@p[n] = i@i;)
            n++;
        }
    }
    @for(i, 0, NDIM, outs[@i]->set_shape({n});)
}
#endif // JIT

} // jittor
