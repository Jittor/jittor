#include "var.h"
#include "hccl_all_reduce_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"

namespace jittor {

#ifndef JIT

static auto hccl_all_reduce = 
    get_op_info("hccl_all_reduce").get_constructor<VarPtr, Var*, string>();

HcclAllReduceOp::HcclAllReduceOp(Var* x, string reduce_op) : x(x), reduce_op(reduce_op) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclAllReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr HcclAllReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return hccl_all_reduce(dout, reduce_op);
}

void HcclAllReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Op:" << reduce_op;
}

#else // JIT

void HcclAllReduceOp::jit_run() {
    //LOGir << "HcclAllReduceOp::jit_run";
    @define(T_HCCL,
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, HcclDataType::HCCL_DATA_TYPE_FP32)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, HcclDataType::HCCL_DATA_TYPE_INT32)
        @if(@strcmp(@Tx,float64)==0, HcclDataType::HCCL_DATA_TYPE_FP64)
        @if(@strcmp(@Tx,int64)==0, HcclDataType::HCCL_DATA_TYPE_INT64)
        @if(@strcmp(@Tx,uint8)==0, HcclDataType::HCCL_DATA_TYPE_UINT8)
        @if(@strcmp(@Tx,float16)==0, HcclDataType::HCCL_DATA_TYPE_FP16)
    )
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
    HCCLCHECK(HcclAllReduce(xp, yp, (uint64_t)x->num, @T_HCCL, @REDUCE_OP, comm, aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
