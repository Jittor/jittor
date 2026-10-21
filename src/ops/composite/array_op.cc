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
    // Gate on `output->num == 1` -- the permanent condition the constructor
    // itself used to decide _force_fuse in the first place -- NOT on
    // _force_fuse directly: fuser.cc's count_fuse() can later clear
    // _force_fuse on a var whose consumers don't all agree on a fused
    // group (see fuser.cc's "only stays forced while every consumer..."
    // comment), and this jit_prepare contribution is this op's ONLY
    // dtype-discriminating input to the jit_key. Two structurally identical
    // size-1 FusedOps wrapping scalar ArrayOps of DIFFERENT dtypes (e.g. a
    // float32 gradient seed from one jt.grad() call and an unrelated
    // float64 seed from a later, independent call) produced byte-identical
    // jit_keys whenever _force_fuse had been cleared this way, so the
    // second call's FusedOp reused the first's cached compiled entry
    // (jit_fused_ops is a plain string-keyed map) -- silently executing the
    // float64 scalar through a kernel compiled for float32 data. Recomputing
    // `num == 1` here instead of reading the (possibly-since-cleared) flag
    // keeps this encoding present regardless of what fuser.cc did to
    // _force_fuse afterward.
    if (output->num == 1) {
        jk << "«T:" << output->dtype();

        // fill or find cbuffer for const var pass
        if (output->dtype().dsize() == 4) {
            auto x = std::abs(ptr<int32>()[0]);
            auto y = std::abs(ptr<float32>()[0]);
            auto z = ptr<uint32>()[0];
            if ((x<=2) || (y==1.0f || y==2.0f))
                jk << "«o:" << z;
        } else if (output->dtype().dsize() == 8) {
            // Same const-buffer fast path as the dsize()==4 branch above,
            // extended to 8-byte scalars (float64/int64), which this
            // branch previously excluded entirely -- excluding them isn't
            // unsafe by itself (the "T:" dtype tag above already guarantees
            // dtype-safety on its own), but it does mean two DIFFERENT
            // 8-byte scalar values of the SAME dtype get no jit_key
            // differentiation from this op at all. z is the low 32 bits
            // only; that's fine, it's just a filter for "is this among the
            // handful of very common small constants".
            auto x = std::abs(ptr<int64>()[0]);
            auto y = std::abs(ptr<float64>()[0]);
            auto z = ptr<uint32>()[0];
            if ((x<=2) || (y==1.0 || y==2.0))
                jk << "«o:" << z << ":" << ptr<uint32>()[1];
        }
        // end of fill cbuffer
    }
}

void ArrayOp::run() {
    // Run again, by a kept graph (`keep_graph`): the first run already handed
    // the data to the output, which still holds it -- see
    // `release_kept_storage`, which leaves a constant's memory alone. Running
    // on would put a null pointer in the output.
    if (!allocation.ptr) {
        ASSERT(output->mem_ptr) << "a re-run array op has lost its data";
        return;
    }
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
