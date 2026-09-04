// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_reduce_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT
NcclReduceOp::NcclReduceOp(Var* x, int root, int group_id)
    : x(x), root(root), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr NcclReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto nccl_broadcast =
        op_constructor<VarPtr, Var*, int, int>("nccl_broadcast");
    return nccl_broadcast(dout, root, group_id);
}

void NcclReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclReduceOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclReduce(
        xp, yp, y->num, nccl_dtype(x->dtype()), ncclSum, root,
        nccl_process_group_comm(group_id), stream));
    // See mpi_reduce_op.cc for why the non-root output stays full-size and
    // deterministic rather than being shrunk or aliased away: every rank must
    // run the same graph. Its contents are meaningless by definition; zero is
    // a filler, not a value.
    //
    // Zeroed on the communication stream rather than the default one so it
    // stays ordered with the collective under either join policy. On the
    // default stream it was only ordered because nccl_stream_end() happened to
    // join right above it, which stops being true inside a deferred bucket.
    if (root != nccl_process_group_rank(group_id))
        checkCudaErrors(cudaMemsetAsync(yp, 0, y->size, stream));
    nccl_stream_end(x, y);
}

#endif
#endif // JIT

} // jittor
