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
#include <algorithm>

namespace jittor {

static bool non_overlapping_dense(Var* value) {
    vector<int> order;
    for (int i=0; i<value->shape.size(); ++i)
        if (value->shape[i] > 1) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return value->storage_stride(a) < value->storage_stride(b);
    });
    int64 extent = 1;
    for (int axis : order) {
        if (value->storage_stride(axis) != extent) return false;
        extent *= value->shape[axis];
    }
    return true;
}

CopyOp::CopyOp(Var* x) : CopyOp(x, false) {}

CopyOp::CopyOp(Var* x, bool preserve) : preserve_format(preserve) {
    if ((!preserve || !non_overlapping_dense(x)) && !x->is_contiguous()) {
        // The original one-argument copy materializes logical order. Preserve
        // that contract and use the existing contiguous owner for holes,
        // overlap and expanded inputs even when format preservation is asked.
        static auto make_copy = op_constructor<VarPtr, Var*>("copy");
        auto dense = contiguous_storage(x);
        forward(make_copy(dense.ptr));
        return;
    }
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
    auto* x = inputs().front();
    auto* y = outputs().front();
    y->set_shape(x->shape);
    if (preserve_format) y->storage_strides = x->storage_strides;
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
