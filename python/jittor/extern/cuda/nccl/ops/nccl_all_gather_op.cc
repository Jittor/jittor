// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_all_gather_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT

static auto nccl_reduce_scatter =
    get_op_info("nccl_reduce_scatter").get_constructor<VarPtr, Var*>();

NcclAllGatherOp::NcclAllGatherOp(Var* x) : x(x) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclAllGatherOp::infer_shape() {
    NanoVector yshape;
    yshape.push_back(mpi_world_size * x->shape[0]);
    for (int i=1; i<x->shape.size(); i++)
        yshape.push_back(x->shape[i]);
    y->set_shape(yshape);
}

VarPtr NcclAllGatherOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nccl_reduce_scatter(dout);
}

void NcclAllGatherOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclAllGatherOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    checkCudaErrors(ncclAllGather(xp, yp, x->num, nccl_dtype(x->dtype()), comm, 0));
}

#endif
#endif // JIT

} // jittor
