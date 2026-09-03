// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.  
//     Guoye Yang <498731903@qq.com>. 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "nccl_broadcast_op.h"
#include "utils/str_utils.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "nccl_wrapper.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT
NcclBroadcastOp::NcclBroadcastOp(Var* x, int root, int group_id)
    : x(x), root(root), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void NcclBroadcastOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr NcclBroadcastOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto nccl_reduce =
        op_constructor<VarPtr, Var*, int, int>("nccl_reduce");
    return nccl_reduce(dout, root, group_id);
}

void NcclBroadcastOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT
#ifdef JIT_cuda

void NcclBroadcastOp::jit_run() {
    // dtype -> ncclDataType_t goes through the single table in
    // nccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    checkCudaErrors(ncclBroadcast(
        xp, yp, y->num, nccl_dtype(x->dtype()), root,
        nccl_process_group_comm(group_id), 0));
}

#endif
#endif // JIT

} // jittor
