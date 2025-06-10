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
    get_op_info("hccl_all_gather").get_constructor<VarPtr, Var*>();

HcclAllGatherOp::HcclAllGatherOp(Var* x) : x(x) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclAllGatherOp::infer_shape() {
    NanoVector yshape;
    yshape.push_back(mpi_world_size * x->shape[0]);
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
    @define(T_HCCL,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, HcclDataType::HCCL_DATA_TYPE_FP32)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, HcclDataType::HCCL_DATA_TYPE_INT32)
        @if(@strcmp(@Tx,float64)==0, HcclDataType::HCCL_DATA_TYPE_FP64)
        @if(@strcmp(@Tx,int64)==0, HcclDataType::HCCL_DATA_TYPE_INT64)
        @if(@strcmp(@Tx,uint8)==0, HcclDataType::HCCL_DATA_TYPE_UINT8)
        @if(@strcmp(@Tx,float16)==0, HcclDataType::HCCL_DATA_TYPE_FP16)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
    HCCLCHECK(HcclAllGather(xp, yp, (uint64_t)x->num, @T_HCCL, comm, aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
