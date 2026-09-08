// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/op_register.h"
#include "ops/composite/copy_op.h"
#include "runtime/backend.h"

namespace jittor {

CopyOp::CopyOp(Var* x) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    auto y = create_output(nullptr, x->dtype());
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr CopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return dout;
}

void CopyOp::infer_shape() {
    outputs().front()->set_shape(inputs().front()->shape);
}

void CopyOp::run() {
    auto x = inputs().front();
    auto size = x->size;
    auto x_ptr = x->mem_ptr;
    auto y_ptr = outputs().front()->mem_ptr;
    #ifdef HAS_ACCELERATOR
    if (executes_on_accelerator()) {
        auto target = allocation_device(outputs().front()->allocator);
        backend_copy_async(y_ptr, target, x_ptr, allocation_device(x->allocator),
            size, backend_stream(target, BackendStreamKind::Compute));
    } else
    #endif
    {
        std::memcpy(y_ptr, x_ptr, size);
    }
}


} // jittor
