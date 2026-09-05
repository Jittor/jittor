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
#include "runtime/device.h"
#include "misc/cuda_streams.h"
#include "mem/swap.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

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
    if (device < 0) {
        if (!y->allocator->is_cuda()) {
            std::memcpy(y->mem_ptr, x->mem_ptr, x->size);
            return;
        }
        Allocation host(cpu_allocator, y->size);
        #ifdef HAS_CUDA
        if (x->allocator->is_cuda())
            checkCudaErrors(cudaMemcpy(host.ptr, x->mem_ptr, x->size,
                                       cudaMemcpyDeviceToHost));
        else
        #endif
            std::memcpy(host.ptr, x->mem_ptr, x->size);

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
    #ifdef HAS_CUDA
    if (runtime_use_cuda()) {
        int src = x->allocator ? x->allocator->device() : -1;
        int dst = y->allocator ? y->allocator->device() : device;
        if (src < 0) {
            // Host-resident source: a plain upload onto the target device,
            // which the executor has already made current.
            checkCudaErrors(cudaMemcpy(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyHostToDevice));
            return;
        }
        if (src == dst) {
            auto stream = cuda_side_stream(CUDA_COPY_STREAM, dst);
            cuda_side_stream_wait_default(CUDA_COPY_STREAM, dst, src);
            checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size,
                                           cudaMemcpyDeviceToDevice, stream));
            cuda_default_stream_wait_side(CUDA_COPY_STREAM, dst, dst);
            return;
        }
        enable_peer_access(src, dst);
        // The destination copy stream waits for the source's computation;
        // both default streams then wait for the copy before either side can
        // consume the result or reuse the source block.
        auto stream = cuda_side_stream(CUDA_COPY_STREAM, dst);
        cuda_side_stream_wait_default(CUDA_COPY_STREAM, dst, src);
        set_current_device(dst);
        checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size,
                                       cudaMemcpyDefault, stream));
        cuda_default_stream_wait_side(CUDA_COPY_STREAM, dst, dst);
        cuda_default_stream_wait_side(CUDA_COPY_STREAM, dst, src);
        return;
    }
    #endif
    std::memcpy(y->mem_ptr, x->mem_ptr, x->size);
}

VarPtr device_copy(Var* x, int device) {
    return make_device_copy(x, device);
}

} // jittor
