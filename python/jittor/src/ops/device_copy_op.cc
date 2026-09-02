// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstring>
#include "var.h"
#include "mem/allocator.h"
#include "ops/op_register.h"
#include "ops/device_copy_op.h"
#include "misc/cuda_flags.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {


DeviceCopyOp::DeviceCopyOp(Var* x, int device) : device(device) {
    CHECK(device >= 0 && (get_device_count() == 0 || device < get_device_count()))
        << "cuda:" << device << "is not a visible device," << get_device_count() << "visible";
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_manual_set_vnbb);
    flags.set(NodeFlags::_cross_device);
    auto y = create_output(nullptr, x->dtype());
    y->cuda_device = device;
    if (x->name.ptr)
        y->name = x->name;
}

VarPtr DeviceCopyOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Resolved lazily: op registration order at static-init time is not defined.
    static auto make_device_copy = get_op_info("device_copy")
        .get_constructor<VarPtr, Var*, int>();
    return make_device_copy(dout, v->cuda_device);
}

void DeviceCopyOp::infer_shape() {
    outputs().front()->set_shape(inputs().front()->shape);
    outputs().front()->cuda_device = device;
}

void DeviceCopyOp::run() {
    auto x = inputs().front();
    auto y = outputs().front();
    #ifdef HAS_CUDA
    if (flags.get(NodeFlags::_cuda)) {
        int src = x->allocator->device;
        if (src >= 0 && src != device)
            checkCudaErrors(cudaMemcpyPeerAsync(y->mem_ptr, device, x->mem_ptr, src, x->size, 0));
        else
            checkCudaErrors(cudaMemcpyAsync(y->mem_ptr, x->mem_ptr, x->size, cudaMemcpyDeviceToDevice, 0));
        return;
    }
    #endif
    std::memcpy(y->mem_ptr, x->mem_ptr, x->size);
}

} // jittor
