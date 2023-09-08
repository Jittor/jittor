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

static auto nccl_all_gather = 
    get_op_info("nccl_all_gather").get_constructor<VarPtr, Var*>();

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
    LOGf << "not implemented";
    return nullptr;
}

void NcclAllGatherOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclAllGatherOp::jit_run() {
    @define(T_NCCL,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, ncclFloat)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, ncclInt)
        @if(@strcmp(@Tx,float64)==0, ncclFloat64)
        @if(@strcmp(@Tx,int64)==0, ncclInt64)
        @if(@strcmp(@Tx,uint8)==0, ncclUint8)
        @if(@strcmp(@Tx,float16)==0, ncclHalf)
        @if(@strcmp(@Tx,bfloat16)==0, ncclBfloat16)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    checkCudaErrors(ncclAllGather(xp, yp, x->num, @T_NCCL, comm, 0));
}

#endif
#endif // JIT

} // jittor
