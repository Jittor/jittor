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
#include "mpi_broadcast_op.h"
#include "ops/op_register.h"
#include "misc/str_utils.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT
MpiBroadcastOp::MpiBroadcastOp(Var* x, int root) : x(x), root(root) {
    #ifdef HAS_CUDA
    if (use_cuda) {
        static VarPtr(*nccl_broadcast)(Var*, int) = nullptr;
        if (!nccl_broadcast && has_op("nccl_broadcast")) {
            nccl_broadcast = get_op_info("nccl_broadcast")
                .get_constructor<VarPtr, Var*, int>();
        }
        if (nccl_broadcast) {
            LOGr << "nccl";
            auto var = nccl_broadcast(x, root);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
    ASSERT(x->dtype().is_float());
}

void MpiBroadcastOp::infer_shape() {
    y->set_shape(x->shape);
}

void MpiBroadcastOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
}

#else // JIT
#ifdef JIT_cpu
void MpiBroadcastOp::jit_run() {
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    int size = 1 @for(i, 0, XDIM,  * xshape@{i});
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    if (mpi_world_rank == root) {
        for (int i = 0; i < mpi_world_size; i++) {
            MPI_Send(xp, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Recv(yp, size, MPI_FLOAT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
#else
void MpiBroadcastOp::jit_run() {
    // cuda device code
}
#endif // JIT_cpu
#endif // JIT

} // jittor
