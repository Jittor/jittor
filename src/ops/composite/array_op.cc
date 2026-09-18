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
    // Through `free_var_mem`, not a hand-rolled copy of it. This used to read
    // `o->mem_ptr`, `o->allocation` and `o->allocator`, free them, and only
    // then overwrite the three fields -- so between the read and the free the
    // var still named an allocation this thread had already given back, and a
    // release on another thread freed the same id a second time. Pinned by the
    // id-space event log (KI-EXEC-007): `set_occupied(16 MiB, thread A)`,
    // `erase_occupied(16 MiB, thread B)`, the id reissued to another block,
    // and then A's free of the same id reporting "allocation not found".
    // `free_var_mem` clears the three fields *before* it calls the allocator,
    // so a second release finds nothing to give back, and it is where the
    // share-ring unlink and the swap path already live.
    //
    // The guard is still needed: this op's output is *created* here, and
    // `create_output` gives it a shape and dtype without an allocation (the
    // `_force_fuse`/scalar shapes take the element path and never get one).
    // Calling `o->allocator->free(...)` on that null allocator is a null
    // dereference -- a loader-bound `jt.array` segfaulted at address 0 inside
    // this function, and the same null storage reaching a copy is the device-1
    // illegal address during the TP weight load.
    if (save_mem || (o->allocator && o->mem_ptr))
        free_var_mem(o);

    o->mem_ptr = allocation.ptr;
    allocation.ptr = nullptr;
    o->allocator = allocation.allocator;
    o->allocation = allocation.allocation;
    if (save_mem) registe_swap(o);
}

} // jittor
