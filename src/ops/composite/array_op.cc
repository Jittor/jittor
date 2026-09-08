// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_ACCELERATOR
#include "mem/allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "core/event_queue.h"
#endif
#include <cstring>
#include <cmath>
#include "core/var.h"
#include "ops/composite/array_op.h"
#include "runtime/device.h"
#include "runtime/backend_streams.h"
#include "runtime/backend.h"
#include "mem/allocator.h"
#include "mem/swap.h"

namespace jittor {

ArrayOp::ArrayOp(const void* ptr, NanoVector shape, NanoString dtype)
    : ArrayOp(ArrayArgs{ptr, shape, dtype}) {}

ArrayOp::ArrayOp(ArrayArgs&& args) {
    output = create_output(args.shape, args.dtype);
    NanoVector shape = output->shape;
    if (output->num == 1) {
        output->set_flag(VarFlags::_force_fuse);
        set_type(OpType::element);
    }
    if (shape.size() == 0)
        output->set_flag(VarFlags::_is_scalar);
    #ifdef HAS_ACCELERATOR
    // Fused scalar values are emitted inside generated kernels on both backends.
    if (requested_backend() != BackendId::Cpu && output->flag(VarFlags::_force_fuse))
        set_flag(OpFlags::_cuda, 1);
    if (requested_backend() != BackendId::Cpu && !save_mem && !use_pinned_host_memory()) {
        set_flag(OpFlags::_cpu, 0);
        set_flag(OpFlags::_cuda, 1);
        if (!output->flag(VarFlags::_force_fuse)) {
            // free prev allocation first
            event_queue.flush();
            // alloc new allocation
            auto size = output->size;
            new (&allocation) Allocation(&cuda_dual_allocator, size);
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
            backend_copy(host_ptr, {}, args.ptr, {}, output->size);
            return;
        }
    }
    #endif
    // TODO: args.buffer too many copy
    new (&allocation) Allocation(get_array_host_allocator(), output->size);
    backend_copy(allocation.ptr, {}, args.ptr, {}, output->size);
}

void ArrayOp::jit_prepare(JK& jk) {
    if (output->flag(VarFlags::_force_fuse)) {
        jk << "«T:" << output->dtype();

        // fill or find cbuffer for const var pass
        if (output->dtype().dsize() == 4) {
            auto x = std::abs(ptr<int32>()[0]);
            auto y = std::abs(ptr<float32>()[0]);
            auto z = ptr<uint32>()[0];
            if ((x<=2) || (y==1.0f || y==2.0f))
                jk << "«o:" << z;
        }
        // end of fill cbuffer
    }
}

void ArrayOp::run() {
    #ifdef HAS_ACCELERATOR
    if (allocation.allocator == &cuda_dual_allocator) {
        auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
        int device = output->device_id;
        auto copy_stream = backend_stream(
            {accelerator_backend_id(), device}, BackendStreamKind::Copy);
        Device target{accelerator_backend_id(), cuda_dual_device_allocator.device()};
        backend_copy_async(allocation.ptr, target, host_ptr, {}, allocation.size,
                           copy_stream);
        backend_default_stream_wait_side(BackendStreamKind::Copy, device, device);
        // delay free this allocation
        allocation.allocator = &delay_free;
    }
    #endif
    // free prev allocation and move into it
    auto o = output;
    // This replaces the output's memory without going through free_var_mem,
    // so the share ring has to be told: whatever o was a sub-range of, it is
    // not one any more (see share_group_link in var.cc).
    if (PREDICT_BRANCH_NOT_TAKEN(o->share_next != nullptr))
        share_group_unlink(o);
    if (save_mem)
        free_with_swap(o);
    else
        o->allocator->free(o->mem_ptr, o->size, o->allocation);
    
    o->mem_ptr = allocation.ptr;
    allocation.ptr = nullptr;
    o->allocator = allocation.allocator;
    o->allocation = allocation.allocation;
    if (save_mem) registe_swap(o);
}

} // jittor
