// ***************************************************************
// Copyright (c) 2021 
//     Guowei Yang <471184555@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mpi_warper.h"
#include "var.h"
#include "mpi_broadcast_op.h"
#include "ops/op_register.h"
#include "misc/str_utils.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT
MpiBroadcastOp::MpiBroadcastOp(Var* x, int root) : x(x), root(root) {
    if (!mpi_enabled) {
        forward(x);
        return;
    }
    #ifdef HAS_CUDA
    if (use_cuda) {
        static auto nccl_broadcast = has_op("nccl_broadcast")
            ? get_op_info("nccl_broadcast").get_constructor<VarPtr, Var*, int>()
            : nullptr;
        if (nccl_broadcast) {
            auto var = nccl_broadcast(x, root);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
}

void MpiBroadcastOp::infer_shape() {
    y->set_shape(x->shape);
    if (root == mpi_world_rank)
        y->share_with(x);
}

VarPtr MpiBroadcastOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto mpi_reduce = 
        get_op_info("mpi_reduce").get_constructor<VarPtr, Var*, NanoString, int>();
    return mpi_reduce(dout, ns_add, root);
}

void MpiBroadcastOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype() << ']';
}

#else // JIT
#ifdef JIT_cpu
void MpiBroadcastOp::jit_run() {
    @define(T_MPI,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, MPI_FLOAT)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, MPI_INT)
        @if(@strcmp(@Tx,float64)==0 || @strcmp(@Tx,double)==0, MPI_DOUBLE)
        @if(@strcmp(@Tx,int64)==0, MPI_DOUBLE_INT)
    )
    auto* __restrict__ yp = y->ptr<Tx>();
    MPI_Bcast(yp, y->num, T_MPI, root, MPI_COMM_WORLD);
}
#endif // JIT_cpu
#endif // JIT

} // jittor
