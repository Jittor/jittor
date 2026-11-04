// ***************************************************************
// Copyright (c) 2026 Jittor.
// All Rights Reserved.
// ***************************************************************
#include "core/var.h"
#include "nccl_send_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT
NcclSendOp::NcclSendOp(Var* x, int peer, int tag, int group_id)
    : x(x), peer(peer), tag(tag), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    int size = nccl_process_group_size(group_id);
    int local_rank = nccl_process_group_rank(group_id);
    USER_CHECKop(peer,>=,0) << "nccl_send peer must be non-negative, got" << peer;
    USER_CHECKop(peer,<,size) << "nccl_send peer" << peer
        << "is outside process group" << group_id << "of size" << size;
    USER_CHECKop(peer,!=,local_rank)
        << "nccl_send cannot target the calling rank" << local_rank;
    USER_CHECK(x->is_contiguous())
        << "nccl_send requires contiguous storage; call contiguous() first";
    // The marker is one element of the payload dtype. It is never read; it
    // exists so the lazy graph has a root to sync and so the bucket scope can
    // reserve it alongside the payload until ncclGroupEnd.
    y = create_output(NanoVector(), x->dtype());
}

void NcclSendOp::infer_shape() {
    // create_output() already gave the marker its 0-dim shape.
}

void NcclSendOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclSendOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see runtime/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclSend(
        xp, (size_t)x->num, nccl_dtype(x->dtype()), peer,
        nccl_process_group_comm(group_id), stream));
    nccl_stream_end(x);
}

#endif
#endif // JIT

} // jittor
