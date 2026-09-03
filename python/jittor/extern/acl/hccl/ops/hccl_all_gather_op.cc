// ***************************************************************
// Copyright (c) 2025 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Jiapeng Zhang <zjp24@mails.tsinghua.edu.cn>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "hccl_all_gather_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"

namespace jittor {

#ifndef JIT

static auto hccl_all_gather =
    op_constructor<VarPtr, Var*, int>("hccl_all_gather");

HcclAllGatherOp::HcclAllGatherOp(Var* x, int group_id)
    : x(x), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclAllGatherOp::infer_shape() {
    NanoVector yshape;
    yshape.push_back(hccl_process_group_size(group_id) * x->shape[0]);
    for (int i=1; i<x->shape.size(); i++)
        yshape.push_back(x->shape[i]);
    y->set_shape(yshape);
}

VarPtr HcclAllGatherOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    LOGf << "not implemented";
    return nullptr;
}

void HcclAllGatherOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT

void HcclAllGatherOp::jit_run() {
    LOGir << "HcclAllGatherOp::jit_run";
    // dtype -> HcclDataType goes through the single table in
    // hccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
    HCCLCHECK(HcclAllGather(
        xp, yp, (uint64_t)x->num, hccl_dtype(x->dtype()),
        hccl_process_group_comm(group_id), aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
