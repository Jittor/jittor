// ***************************************************************
// Copyright (c) 2021 
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_reduce_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_warper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT
NcclReduceOp::NcclReduceOp(Var* x, int root) : x(x), root(root) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr NcclReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto nccl_broadcast = 
        get_op_info("nccl_broadcast").get_constructor<VarPtr, Var*, int>();
    return nccl_broadcast(dout,root);
}

void NcclReduceOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype() << ']';
}

#else // JIT
#ifdef JIT_cuda

void NcclReduceOp::jit_run() {
    @define(T_NCCL,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, ncclFloat)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, ncclInt)
        @if(@strcmp(@Tx,float64)==0, ncclFloat64)
        @if(@strcmp(@Tx,int64)==0, ncclInt64)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    checkCudaErrors(ncclReduce(xp, yp, y->num, @T_NCCL, ncclSum, root, comm, 0));
    if (root != mpi_world_rank)
        checkCudaErrors(cudaMemsetAsync(yp, 0, y->size));
}

#endif
#endif // JIT

} // jittor
