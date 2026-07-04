// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#include "var.h"
#include "nccl_reduce_scatter_op.h"
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

NcclReduceScatterOp::NcclReduceScatterOp(Var* x) : x(x) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclReduceScatterOp::infer_shape() {
    NanoVector yshape;
    CHECKop(x->shape.size(),>=,1);
    CHECKop(x->shape[0] % mpi_world_size,==,0)
        << "nccl_reduce_scatter expects dim0 divisible by world size";
    yshape.push_back(x->shape[0] / mpi_world_size);
    for (int i=1; i<x->shape.size(); i++)
        yshape.push_back(x->shape[i]);
    y->set_shape(yshape);
}

VarPtr NcclReduceScatterOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nccl_all_gather(dout);
}

void NcclReduceScatterOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclReduceScatterOp::jit_run() {
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
    checkCudaErrors(ncclReduceScatter(xp, yp, y->num, @T_NCCL, ncclSum, comm, 0));
}

#endif
#endif // JIT

} // jittor
