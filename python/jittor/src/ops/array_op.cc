// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "mem/allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include <cstring>
#include <cmath>
#include "var.h"
#include "ops/array_op.h"
#include "misc/cuda_flags.h"
#include "misc/cuda_streams.h"
#include "mem/allocator.h"
#include "mem/swap.h"

namespace jittor {

#ifdef IS_ACL
#pragma GCC visibility push(hidden)
namespace array_local {
cudaStream_t stream;
cudaEvent_t event;
// A stream and an event belong to the device they were created on: an event
// of device 0 cannot be recorded on device 1's default stream. So there is a
// pair per device and the globals name the current device's, swapped by the
// device-switch hook -- the same shape as the library handles.
static vector<cudaStream_t> streams;
static vector<cudaEvent_t> events;

static void array_switch_device(int device) {
    if ((int)streams.size() <= device) {
        streams.resize(device+1, nullptr);
        events.resize(device+1, nullptr);
    }
    if (!streams[device]) {
        checkCudaErrors(cudaStreamCreateWithFlags(&streams[device], cudaStreamNonBlocking));
        checkCudaErrors(cudaEventCreate(&events[device], cudaEventDisableTiming));
    }
    stream = streams[device];
    event = events[device];
}

struct Init {
Init() {
    if (!get_device_count()) return;
    add_device_switch_hook(array_switch_device);
}
~Init() {
    if (!get_device_count()) return;
    peekCudaErrors(cudaDeviceSynchronize());
    for (size_t i=0; i<streams.size(); i++)
        if (streams[i]) {
            peekCudaErrors(cudaStreamDestroy(streams[i]));
            peekCudaErrors(cudaEventDestroy(events[i]));
        }
    streams.clear();
    events.clear();
}
} init;

}
using namespace array_local;

#endif

ArrayOp::ArrayOp(const void* ptr, NanoVector shape, NanoString dtype)
    : ArrayOp(ArrayArgs{ptr, shape, dtype}) {}

DECLARE_FLAG(int, use_cuda_host_allocator);

ArrayOp::ArrayOp(ArrayArgs&& args) {
    output = create_output(args.shape, args.dtype);
    NanoVector shape = output->shape;
    if (shape.size() == 1 && shape[0] == 1) {
        output->set_flag(VarFlags::_force_fuse);
        output->set_flag(VarFlags::_is_scalar);
        set_type(OpType::element);
    }
    #ifdef HAS_CUDA
    // Fused scalar values are emitted inside generated kernels on both backends.
    if (use_cuda && output->flag(VarFlags::_force_fuse))
        set_flag(OpFlags::_cuda, 1);
    if (use_cuda && !save_mem && !use_cuda_host_allocator) {
        set_flag(OpFlags::_cpu, 0);
        set_flag(OpFlags::_cuda, 1);
        if (!output->flag(VarFlags::_force_fuse)) {
            // free prev allocation first
            event_queue.flush();
            // alloc new allocation
            auto size = output->size;
            new (&allocation) Allocation(&cuda_dual_allocator, size);
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
            std::memcpy(host_ptr, args.ptr, output->size);
            return;
        }
    }
    #endif
    // TODO: args.buffer too many copy
    new (&allocation) Allocation(cpu_allocator, output->size);
    std::memcpy(allocation.ptr, args.ptr, output->size);
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
    #ifdef HAS_CUDA
    if (allocation.allocator == &cuda_dual_allocator) {
        auto host_ptr = cuda_dual_allocator.get_dual_allocation(allocation.allocation).host_ptr;
        #ifdef IS_CUDA
        int device = output->device_id;
        auto copy_stream = cuda_side_stream(CUDA_COPY_STREAM, device);
        checkCudaErrors(cudaMemcpyAsync(
            allocation.ptr, host_ptr, allocation.size, cudaMemcpyHostToDevice,
            copy_stream));
        cuda_default_stream_wait_side(CUDA_COPY_STREAM, device, device);
        #else
        checkCudaErrors(cudaMemcpyAsync(
            allocation.ptr, host_ptr, allocation.size, cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaEventRecord(event, stream));
        // ACL kernels run on aclstream rather than CUDA's default stream.
        checkCudaErrors(cudaStreamWaitEvent(aclstream, event, 0));
        #endif
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
