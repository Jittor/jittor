#include "var.h"
#include "hccl_reduce_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"

namespace jittor {

#ifndef JIT

HcclReduceOp::HcclReduceOp(Var* x, string reduce_op, int root) : x(x), reduce_op(reduce_op), root(root) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclReduceOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr HcclReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    static auto hccl_broadcast = 
        get_op_info("hccl_broadcast").get_constructor<VarPtr, Var*, int>();
    return hccl_broadcast(dout, root);
}

void HcclReduceOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Op:" << reduce_op;
    jk << "«Root:" << root;
}

#else // JIT

void HcclReduceOp::jit_run() {
    LOGir << "HcclReduceOp::jit_run";
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
    HCCLCHECK(HcclReduce(xp, yp, (uint64_t)x->num, hccl_dtype(x->dtype()), @REDUCE_OP, @Root, comm, aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
