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

// Resolved on first use rather than at static-init time: the order in which
// the op registry is filled across translation units is not defined.
static VarPtr make_device_copy(Var* x, int device) {
    static auto ctor = get_op_info("device_copy")
        .get_constructor<VarPtr, Var*, int>();
    return ctor(x, device);
}

DeviceCopyOp::DeviceCopyOp(Var* x, int device) : x(x), device(device) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_manual_set_vnbb);
    // This is the one op whose output device is not its input's, so
    // Op::init must leave the placement alone.
    flags.set(NodeFlags::_manual_device);
    int count = get_device_count();
    CHECK(device >= 0 && (count == 0 || device < count))
        << "Invalid CUDA device index" << device >> ", visible device count is" << count;
    y = create_output(nullptr, x->dtype());
    y->device_id = device;
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr DeviceCopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // The gradient of a move is a move back.
    return make_device_copy(dout, x->device_id);
}

void DeviceCopyOp::infer_shape() {
    y->set_shape(x->shape);
    y->device_id = device;
}

void DeviceCopyOp::jit_prepare(JK& jk) {
    // No generated kernel: run() issues the copy itself.
}

#ifdef HAS_CUDA
// One reusable event per device, recorded on that device's default stream.
// An event belongs to the device it was created on, so it is created with
// that device current.
static cudaEvent_t device_event(int device) {
    static vector<cudaEvent_t> events;
    if ((int)events.size() <= device) events.resize(device+1, nullptr);
    if (!events[device]) {
        int prev = current_device();
        if (prev != device) set_current_device(device);
        checkCudaErrors(cudaEventCreateWithFlags(&events[device], cudaEventDisableTiming));
        if (prev >= 0 && prev != device) set_current_device(prev);
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
            // Host-resident source: a plain upload onto the target device,
            // which the executor has already made current.
            checkCudaErrors(cudaMemcpy(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyHostToDevice));
            return;
        }
        if (src == dst) {
            checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyDeviceToDevice, 0));
            return;
        }
        enable_peer_access(src, dst);
        // Each device drives its own default stream, and the two are not
        // ordered against each other by anything. So:
        //   1. record on src, after the kernels that produced x;
        //   2. have dst's stream wait for that before the copy -- otherwise
        //      the copy reads x while it is still being written;
        //   3. record on dst after the copy and have src's stream wait for
        //      it, so x's block cannot be recycled into another kernel while
        //      the copy is still reading it.
        // cudaMemcpyPeer would give the ordering for free by being
        // synchronous, at the price of draining both pipelines on every move.
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
