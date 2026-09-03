// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guowei Yang <471184555@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mpi_wrapper.h"
#include "var.h"
#include "mpi_broadcast_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "misc/cuda_flags.h"
#include <cstring>

namespace jittor {

#ifndef JIT
MpiBroadcastOp::MpiBroadcastOp(Var* x, int root) : x(x), root(root) {
    if (!mpi_enabled) {
        forward(x);
        return;
    }
    #ifdef HAS_CUDA
    if (use_device_mpi && use_cuda) {
        static auto nccl_broadcast = has_op("nccl_broadcast")
            ? get_op_info("nccl_broadcast").get_constructor<VarPtr, Var*, int>()
            : nullptr;
        static auto hccl_broadcast = has_op("hccl_broadcast")
            ? get_op_info("hccl_broadcast").get_constructor<VarPtr, Var*, int>()
            : nullptr;
        if (nccl_broadcast) {
            auto var = nccl_broadcast(x, root);
            forward(var);
            return;
        } else if (hccl_broadcast) {
            auto var = hccl_broadcast(x, root);
            //exe.run_sync({var}, true);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
}

void MpiBroadcastOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr MpiBroadcastOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto mpi_reduce = op_constructor<VarPtr, Var*, NanoString, int>("mpi_reduce");
    return mpi_reduce(dout, ns_add, root);
}

void MpiBroadcastOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cpu
void MpiBroadcastOp::jit_run() {
    // dtype -> MPI type goes through the single table in mpi_wrapper.cc
    // (see misc/collective_dtype.h); the copy this replaces mapped int64 to
    // the 16-byte MAXLOC pair MPI_DOUBLE_INT, so a broadcast of n int64
    // elements wrote 2n of them.
    //
    // The root's copy is what `infer_shape` used to avoid with
    // `y->share_with(x)`: y reused x's buffer, so MPI_Bcast sent straight out
    // of it. That saved a memcpy and cost graph isomorphism -- the output was
    // an alias of the input on one rank and a fresh allocation on every other,
    // so the ranks were no longer running the same graph, and the aliasing
    // decision lived in shape inference, which has no business making it. A
    // graph that differs by rank is the kind of defect whose symptom appears
    // nowhere near its cause (rank 0 fuses differently from rank 1), and
    // 8.11's acceptance is that the ranks agree. One host memcpy on the root
    // is the price; this operator's JIT is CPU-only (CUDA goes to
    // nccl_broadcast), and the callers are parameter broadcast at startup and
    // the backward of mpi_reduce.
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    if (root == mpi_world_rank)
        std::memcpy(yp, xp, y->size);
    MPI_CHECK(MPI_Bcast(yp, y->num, mpi_dtype(y->dtype()), root, MPI_COMM_WORLD));
}
#endif // JIT_cpu
#endif // JIT

} // jittor
