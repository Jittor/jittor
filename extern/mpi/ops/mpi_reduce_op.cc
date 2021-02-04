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
#include "mpi_reduce_op.h"
#include "ops/op_register.h"
#include "misc/str_utils.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT

static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_mpi_reduce = get_op_info("mpi_reduce")
    .get_constructor<VarPtr, Var*, NanoString, int>();

MpiReduceOp::MpiReduceOp(Var* x, NanoString op, int root) : x(x), op(op), root(root) {
    if (!mpi_enabled) {
        forward(x);
        return;
    }
    if (op == ns_mean) {
        auto var = make_mpi_reduce(x, ns_add, root);
        var = make_binary(var, make_array(&mpi_world_size, 1, ns_int32), ns_divide);
        forward(var);
        return;
    }
    ASSERT(op == ns_add) << "Not supported MPI op" << op;
    #ifdef HAS_CUDA
    if (use_cuda) {
        static auto nccl_reduce = has_op("nccl_reduce")
            ? get_op_info("nccl_reduce").get_constructor<VarPtr, Var*, int>()
            : nullptr;
        if (nccl_reduce) {
            auto var = nccl_reduce(x, root);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
}

void MpiReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr MpiReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static VarPtr(*mpi_broadcast)(Var*, int) = 
        get_op_info("mpi_broadcast").get_constructor<VarPtr, Var*, int>();
    return mpi_broadcast(dout,root);
}

void MpiReduceOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][OP:") << op << ']';
}

#else // JIT
#ifdef JIT_cpu
void MpiReduceOp::jit_run() {
    @define(T_MPI,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, MPI_FLOAT)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, MPI_INT)
        @if(@strcmp(@Tx,float64)==0 || @strcmp(@Tx,double)==0, MPI_DOUBLE)
        @if(@strcmp(@Tx,int64)==0, MPI_DOUBLE_INT)
    )
    @define(OP_MPI,
        @if(@strcmp(@OP,add)==0, MPI_SUM)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    index_t num = y->num;
    MPI_CHECK(MPI_Reduce(xp, yp, num, T_MPI, OP_MPI, root, MPI_COMM_WORLD));
    if (root != mpi_world_rank)
        for (index_t i=0; i<num; i++) yp[i] = 0;
}
#endif // JIT_cpu
#endif // JIT

} // jittor
