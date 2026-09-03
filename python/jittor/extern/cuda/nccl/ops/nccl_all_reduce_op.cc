// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_all_reduce_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT

static auto nccl_all_reduce =
    op_constructor<VarPtr, Var*, int>("nccl_all_reduce");

NcclAllReduceOp::NcclAllReduceOp(Var* x, int group_id)
    : x(x), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclAllReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr NcclAllReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nccl_all_reduce(dout, group_id);
}

void NcclAllReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclAllReduceOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    auto stream = nccl_stream_begin();
    checkCudaErrors(ncclAllReduce(
        xp, yp, y->num, nccl_dtype(x->dtype()), ncclSum,
        nccl_process_group_comm(group_id), stream));
    nccl_stream_end();
}

#endif
#endif // JIT

} // jittor
