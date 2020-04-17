// ***************************************************************
// Copyright (c) 2020 
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
MpiReduceOp::MpiReduceOp(Var* x, int root) : x(x), root(root) {
    #ifdef HAS_CUDA
    if (use_cuda) {
        static VarPtr(*nccl_reduce)(Var*, int) = nullptr;
        if (!nccl_reduce && has_op("nccl_reduce")) {
            nccl_reduce = get_op_info("nccl_reduce")
                .get_constructor<VarPtr, Var*, int>();
        }
        if (nccl_reduce) {
            LOGr << "nccl";
            auto var = nccl_reduce(x, root);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
    ASSERT(x->dtype().is_float());
}

void MpiReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr MpiReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static VarPtr(*mpi_broadcast)(Var*, int) = 
        get_op_info("mpi_broadcast").get_constructor<VarPtr, Var*, int>();
    return mpi_broadcast(dout,root);
}

void MpiReduceOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
}

#else // JIT
#ifdef JIT_cpu
void MpiReduceOp::jit_run() {
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    int size = 1 @for(i, 0, XDIM,  * xshape@{i});
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    MPI_Reduce(xp, yp, size, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);
}
#else
void MpiReduceOp::jit_run() {
    // cuda device code
}
#endif // JIT_cpu
#endif // JIT

} // jittor
