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
#include "mpi_reduce_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT

static auto make_array = op_constructor<VarPtr, const void*, NanoVector, NanoString>("array");
static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_mpi_reduce = op_constructor<VarPtr, Var*, NanoString, int>("mpi_reduce");

MpiReduceOp::MpiReduceOp(Var* x, NanoString op, int root) : x(x), op(op), root(root) {
    if (!mpi_enabled) {
        forward(x);
        return;
    }
    if (op == ns_mean) {
        auto var = make_mpi_reduce(x, ns_add, root);
        var = make_binary(var, make_array(&mpi_world_size, {}, ns_int32), ns_divide);
        forward(var);
        return;
    }
    ASSERT(op == ns_add) << "Not supported MPI op" << op;
    #ifdef HAS_CUDA
    if (use_device_mpi && runtime_use_cuda()) {
        static auto nccl_reduce = has_op("nccl_reduce")
            ? get_op_info("nccl_reduce").get_constructor<VarPtr, Var*, int, int>()
            : nullptr;
        static auto hccl_reduce = has_op("hccl_reduce")
            ? get_op_info("hccl_reduce").get_constructor<VarPtr, Var*, string, int, int>()
            : nullptr;
        if (nccl_reduce) {
            auto var = nccl_reduce(x, root, 0);
            forward(var);
            return;
        } else if (hccl_reduce) {
            auto var = hccl_reduce(x, "sum", root, 0);
            //runtime_executor().run_sync({var}, true);
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
    jk << "«Tx:" << x->dtype();
    jk << "«OP:" << op;
}

#else // JIT
#ifdef JIT_cpu
void MpiReduceOp::jit_run() {
    // dtype -> MPI type/op goes through the single table in mpi_wrapper.cc
    // (see misc/collective_dtype.h); the copy this replaces mapped int64 to
    // the 16-byte MAXLOC pair MPI_DOUBLE_INT.
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    index_t num = y->num;
    MPI_CHECK(MPI_Reduce(xp, yp, num,
        mpi_dtype(x->dtype()), mpi_add_op(x->dtype()), root, MPI_COMM_WORLD));
    // Only `root` receives the reduction; MPI ignores recvbuf everywhere else,
    // so on any other rank this output holds nothing. It is still allocated at
    // full size and filled deterministically, on purpose:
    //
    // The obvious saving -- give non-root ranks a smaller output, or alias the
    // input -- would make the graph's shape or alias structure depend on the
    // rank. That is the defect 8.11's other half removes from mpi_broadcast,
    // and it is far more expensive than the buffer: ranks that do not share a
    // graph fuse differently, and the symptom then surfaces nowhere near the
    // cause. So every rank keeps the same shape and the same allocation, and
    // "non-root output is meaningless" is a contract stated here and in the
    // Python docstring rather than a shape you have to notice.
    //
    // Zero rather than left-over memory: reading it is a caller bug either
    // way, but a deterministic value makes that bug reproduce identically
    // instead of depending on what the allocator handed back.
    if (root != mpi_world_rank)
        for (index_t i=0; i<num; i++) yp[i] = 0;
}
#endif // JIT_cpu
#endif // JIT

} // jittor
