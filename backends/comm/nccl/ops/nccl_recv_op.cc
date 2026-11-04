// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#include "core/var.h"
#include "nccl_recv_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT
NcclRecvOp::NcclRecvOp(NanoVector shape, NanoString dtype, int src, int tag,
                       int group_id)
    : shape(shape), dtype(dtype), src(src), tag(tag), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    int size = nccl_process_group_size(group_id);
    int local_rank = nccl_process_group_rank(group_id);
    USER_CHECKop(src,>=,0) << "nccl_recv source must be non-negative, got" << src;
    USER_CHECKop(src,<,size) << "nccl_recv source" << src
        << "is outside process group" << group_id << "of size" << size;
    USER_CHECKop(src,!=,local_rank)
        << "nccl_recv cannot source from the calling rank" << local_rank;
    y = create_output(shape, dtype);
}

void NcclRecvOp::infer_shape() {
    // create_output() already set y->shape from the constructor's shape.
}

void NcclRecvOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << y->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclRecvOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see runtime/collective_dtype.h).
    auto* __restrict__ yp = y->ptr<Tx>();
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclRecv(
        yp, (size_t)y->num, nccl_dtype(y->dtype()), src,
        nccl_process_group_comm(group_id), stream));
    // No input Var: only the received buffer has to stay reserved until the
    // bucket's ncclGroupEnd.
    nccl_stream_end(nullptr, y);
}

#endif
#endif // JIT

} // jittor
