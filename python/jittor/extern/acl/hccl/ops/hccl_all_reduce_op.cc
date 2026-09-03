#include "var.h"
#include "hccl_all_reduce_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"

namespace jittor {

#ifndef JIT

static auto hccl_all_reduce =
    op_constructor<VarPtr, Var*, string, int>("hccl_all_reduce");

HcclAllReduceOp::HcclAllReduceOp(Var* x, string reduce_op, int group_id)
    : x(x), reduce_op(reduce_op), group_id(group_id) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclAllReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr HcclAllReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return hccl_all_reduce(dout, reduce_op, group_id);
}

void HcclAllReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Op:" << reduce_op;
}

#else // JIT

void HcclAllReduceOp::jit_run() {
    //LOGir << "HcclAllReduceOp::jit_run";
    // dtype -> HcclDataType goes through the single table in
    // hccl_wrapper.cc (see misc/collective_dtype.h).
    @define(REDUCE_OP,
        @if(@strcmp(@Op,sum)==0, HcclReduceOp::HCCL_REDUCE_SUM)
        @if(@strcmp(@Op,prod)==0, HcclReduceOp::HCCL_REDUCE_PROD)
        @if(@strcmp(@Op,max)==0, HcclReduceOp::HCCL_REDUCE_MAX)
        @if(@strcmp(@Op,min)==0, HcclReduceOp::HCCL_REDUCE_MIN)
    )
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
    HCCLCHECK(HcclAllReduce(
        xp, yp, (uint64_t)x->num, hccl_dtype(x->dtype()), @REDUCE_OP,
        hccl_process_group_comm(group_id), aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
