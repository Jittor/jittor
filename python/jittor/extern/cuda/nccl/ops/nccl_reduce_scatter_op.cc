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
    op_constructor<VarPtr, Var*, int>("nccl_all_gather");

NcclReduceScatterOp::NcclReduceScatterOp(Var* x, int group_id)
    : x(x), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclReduceScatterOp::infer_shape() {
    NanoVector yshape;
    USER_CHECKop(x->shape.size(),>=,1);
    int group_size = nccl_process_group_size(group_id);
    USER_CHECKop(x->shape[0] % group_size,==,0)
        << "nccl_reduce_scatter expects dim0 divisible by process-group size";
    yshape.push_back(x->shape[0] / group_size);
    for (int i=1; i<x->shape.size(); i++)
        yshape.push_back(x->shape[i]);
    y->set_shape(yshape);
}

VarPtr NcclReduceScatterOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nccl_all_gather(dout, group_id);
}

void NcclReduceScatterOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclReduceScatterOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclReduceScatter(
        xp, yp, y->num, nccl_dtype(x->dtype()), ncclSum,
        nccl_process_group_comm(group_id), stream));
    nccl_stream_end();
}

#endif
#endif // JIT

} // jittor
