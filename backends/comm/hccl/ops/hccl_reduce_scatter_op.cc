// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// ***************************************************************
#include "core/var.h"
#include "hccl_reduce_scatter_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"

namespace jittor {

#ifndef JIT

static auto hccl_all_gather =
    op_constructor<VarPtr, Var*, int>("hccl_all_gather");

HcclReduceScatterOp::HcclReduceScatterOp(Var* x, int group_id)
    : x(x), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclReduceScatterOp::infer_shape() {
    NanoVector yshape;
    USER_CHECKop(x->shape.size(),>=,1);
    int group_size = hccl_process_group_size(group_id);
    USER_CHECKop(x->shape[0] % group_size,==,0)
        << "hccl_reduce_scatter expects dim0 divisible by process-group size";
    yshape.push_back(x->shape[0] / group_size);
    for (int i=1; i<x->shape.size(); i++)
        yshape.push_back(x->shape[i]);
    y->set_shape(yshape);
}

VarPtr HcclReduceScatterOp::grad(
        Var* out, Var* dout, Var* v, int v_index) {
    return hccl_all_gather(dout, group_id);
}

void HcclReduceScatterOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
}

#else // JIT

void HcclReduceScatterOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    hccl_collective_begin();
    HCCLCHECK(HcclReduceScatter(
        xp, yp, (uint64_t)y->num, hccl_dtype(x->dtype()),
        HcclReduceOp::HCCL_REDUCE_SUM,
        hccl_process_group_comm(group_id), aclstream));
    hccl_collective_end();
}

#endif // JIT

} // namespace jittor
