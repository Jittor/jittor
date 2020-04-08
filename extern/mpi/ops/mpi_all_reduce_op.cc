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
#include "mpi_all_reduce_op.h"
#include "ops/op_register.h"
#include "misc/str_utils.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT
MpiAllReduceOp::MpiAllReduceOp(Var* x) : x(x) {
    #ifdef HAS_CUDA
    if (use_cuda) {
        static VarPtr(*nccl_all_reduce)(Var*) = nullptr;
        if (!nccl_all_reduce && has_op("nccl_all_reduce")) {
            nccl_all_reduce = get_op_info("nccl_all_reduce")
                .get_constructor<VarPtr, Var*>();
        }
        if (nccl_all_reduce) {
            LOGr << "nccl";
            auto var = nccl_all_reduce(x);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
    ASSERT(x->dtype().is_float());
}

void MpiAllReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

void MpiAllReduceOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
}

#else // JIT
#ifdef JIT_cpu
void MpiAllReduceOp::jit_run() {
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    int size = 1 @for(i, 0, XDIM,  * xshape@{i});
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    MPI_Allreduce(xp, yp, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}
#else
void MpiAllReduceOp::jit_run() {
    // cuda device code
}
#endif // JIT_cpu
#endif // JIT

} // jittor
