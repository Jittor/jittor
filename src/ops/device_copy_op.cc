// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "mem/allocator.h"
#include "ops/device_copy_op.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/backend.h"
#include "mem/swap.h"

namespace jittor {

// Resolved on first use rather than at static-init time: the order in which
// the op registry is filled across translation units is not defined.
static VarPtr make_device_copy(Var* x, int device) {
    static auto ctor = op_constructor<VarPtr, Var*, int>("device_copy");
    return ctor(x, device);
}

DeviceCopyOp::DeviceCopyOp(Var* x, int device) : x(x), device(device) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    // This is the one op whose output device is not its input's, so
    // Op::init must leave the placement alone.
    set_flag(OpFlags::_manual_device);
    int count = get_device_count();
    USER_CHECK(device >= -1 && (device < 0 || count == 0 || device < count))
        << "Invalid CUDA device index" << device >> ", visible device count is" << count;
    y = create_output(nullptr, x->dtype());
    y->device_id = device < 0 ? x->device_id : device;
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr DeviceCopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // The gradient of a move is a move back.
    return make_device_copy(dout, x->device_id);
}

void DeviceCopyOp::infer_shape() {
    y->set_shape(x->shape);
    y->device_id = device < 0 ? x->device_id : device;
}

void DeviceCopyOp::jit_prepare(JK& jk) {
    // No generated kernel: run() issues the copy itself.
}

void DeviceCopyOp::run() {
    auto source = allocation_device(x->allocator);
    if (device < 0) {
        if (!y->allocator->is_cuda()) {
            backend_copy(y->mem_ptr, {}, x->mem_ptr, source, x->size);
            return;
        }
        Allocation host(cpu_allocator, y->size);
        backend_copy(host.ptr, {}, x->mem_ptr, source, x->size);

        // The executor allocates outputs on the op's device before run(). A
        // host copy is the exception: replace that temporary device block
        // with the independently allocated host block after the D2H copy.
        if (save_mem)
            free_with_swap(y);
        else
            y->allocator->free(y->mem_ptr, y->size, y->allocation);
        y->mem_ptr = host.ptr;
        y->allocation = host.allocation;
        y->allocator = host.allocator;
        host.ptr = nullptr;
        if (save_mem) registe_swap(y);
        return;
    }
    auto target = allocation_device(y->allocator);
    // Ordered backend copies retain the source until the transfer completes
    // and make the destination compute stream wait before consuming it.
    backend_copy(y->mem_ptr, target, x->mem_ptr, source, x->size, true);
}

VarPtr device_copy(Var* x, int device) {
    return make_device_copy(x, device);
}

} // jittor
