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
#include "mpi_all_reduce_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "misc/cuda_flags.h"

namespace jittor {

#ifndef JIT

static auto make_array = op_constructor<VarPtr, const void*, NanoVector, NanoString>("array");
static auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
static auto make_mpi_all_reduce = op_constructor<VarPtr, Var*, NanoString>("mpi_all_reduce");

MpiAllReduceOp::MpiAllReduceOp(Var* x, NanoString op) : x(x), op(op) {
    if (!mpi_enabled) {
        forward(x);
        return;
    }
    if (op == ns_mean) {
        auto var = make_mpi_all_reduce(x, ns_add);
        var = make_binary(var, make_array(&mpi_world_size, 1, ns_int32), ns_divide);
        forward(var);
        return;
    }
    ASSERT(op == ns_add) << "Not supported MPI op" << op;
    #ifdef HAS_CUDA

    if (use_device_mpi && use_cuda) {
        static auto nccl_all_reduce = has_op("nccl_all_reduce")
            ? get_op_info("nccl_all_reduce").get_constructor<VarPtr, Var*, int>()
            : nullptr;
        static auto hccl_all_reduce = has_op("hccl_all_reduce")
            ? get_op_info("hccl_all_reduce").get_constructor<VarPtr, Var*, string, int>()
            : nullptr;
        if (nccl_all_reduce) {
            auto var = nccl_all_reduce(x, 0);
            forward(var);
            return;
        } else if (hccl_all_reduce) {
            auto var = hccl_all_reduce(x, "sum", 0);
            //exe.run_sync({var}, true);
            forward(var);
            return;
        }
    }
    #endif
    y = create_output(nullptr, x->dtype());
}

void MpiAllReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr MpiAllReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto mpi_all_reduce = op_constructor<VarPtr, Var*,NanoString>("mpi_all_reduce");
    return mpi_all_reduce(dout, ns_add);
}

void MpiAllReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«OP:" << op;
}

#else // JIT
#ifdef JIT_cpu
void MpiAllReduceOp::jit_run() {
    // dtype -> MPI type/op goes through the single table in mpi_wrapper.cc
    // (see misc/collective_dtype.h). The per-operator table this replaces
    // mapped int64 to MPI_DOUBLE_INT, a 16-byte MAXLOC pair, so `num`
    // elements of it read 2x past the end of x and returned garbage.
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    index_t num = y->num;
    MPI_CHECK(MPI_Allreduce(xp, yp, num,
        mpi_dtype(x->dtype()), mpi_add_op(x->dtype()), MPI_COMM_WORLD));
}
#endif // JIT_cpu
#endif // JIT

} // jittor
