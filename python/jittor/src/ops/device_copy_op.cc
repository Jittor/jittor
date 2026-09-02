// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstring>
#include "var.h"
#include "mem/allocator.h"
#include "ops/device_copy_op.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {

static auto make_device_copy = get_op_info("device_copy")
    .get_constructor<VarPtr, Var*, int>();

DeviceCopyOp::DeviceCopyOp(Var* x, int device) : x(x), device(device) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_manual_set_vnbb);
    flags.set(NodeFlags::_manual_device);
    #ifdef HAS_CUDA
    int count = get_device_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA device index" << device << ", device count is" << count;
    #endif
    y = create_output(nullptr, x->dtype());
    y->device_id = device;
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr DeviceCopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_device_copy(dout, x->device_id);
}

void DeviceCopyOp::infer_shape() {
    y->set_shape(x->shape);
}

void DeviceCopyOp::jit_prepare(JK& jk) {
    // No kernel: run() issues the copy.
}

#ifdef HAS_CUDA
// One reusable event per device, recorded on that device's default stream.
static cudaEvent_t device_event(int device) {
    static vector<cudaEvent_t> events;
    if ((int)events.size() <= device) events.resize(device+1, nullptr);
    if (!events[device]) {
        int prev = current_device();
        if (prev != device) set_current_device(device);
        checkCudaErrors(cudaEventCreateWithFlags(&events[device], cudaEventDisableTiming));
        if (prev != device) set_current_device(prev);
    }
    return events[device];
}
#endif

void DeviceCopyOp::run() {
    #ifdef HAS_CUDA
    if (use_cuda) {
        int src = x->allocator ? x->allocator->device() : -1;
        int dst = y->allocator ? y->allocator->device() : device;
        if (src < 0) {
            // Host-resident source: a plain upload onto the target device.
            checkCudaErrors(cudaMemcpy(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyHostToDevice));
            return;
        }
        if (src == dst) {
            checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyDeviceToDevice, 0));
            return;
        }
        enable_peer_access(src, dst);
        // Each device runs its own default stream. Order the copy after the
        // producer of x on src, issue it on dst, then make src wait for it
        // before anything there may reuse x's memory.
        auto ev_src = device_event(src), ev_dst = device_event(dst);
        set_current_device(src);
        checkCudaErrors(cudaEventRecord(ev_src, 0));
        set_current_device(dst);
        checkCudaErrors(cudaStreamWaitEvent(0, ev_src, 0));
        checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyDefault, 0));
        checkCudaErrors(cudaEventRecord(ev_dst, 0));
        set_current_device(src);
        checkCudaErrors(cudaStreamWaitEvent(0, ev_dst, 0));
        set_current_device(dst);
        return;
    }
    #endif
    std::memcpy(y->mem_ptr, x->mem_ptr, x->size);
}

VarPtr device_copy(Var* x, int device) {
    return make_device_copy(x, device);
}

} // jittor
