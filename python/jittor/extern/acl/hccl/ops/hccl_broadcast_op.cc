#include "var.h"
#include "hccl_broadcast_op.h"
#include "ops/op_register.h"
#include "utils/str_utils.h"
#include "hccl_wrapper.h"
#include <cassert>

namespace jittor {

#ifndef JIT

static auto hccl_broadcast = 
    get_op_info("hccl_broadcast").get_constructor<VarPtr, Var*, int>();

HcclBroadcastOp::HcclBroadcastOp(Var* x, int root) : x(x), root(root) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void HcclBroadcastOp::infer_shape() {
    y->set_shape(x->shape);
}

VarPtr HcclBroadcastOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return hccl_broadcast(dout, root);
}

void HcclBroadcastOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Root:" << root;
}

#else // JIT

void HcclBroadcastOp::jit_run() {
    //LOGir << "HcclBroadcastOp::jit_run";
    // dtype -> HcclDataType goes through the single table in
    // hccl_wrapper.cc (see misc/collective_dtype.h).
    auto* __restrict__ xp = x->ptr<Tx>();
    auto* __restrict__ yp = y->ptr<Tx>();
    //LOGir << "HcclBroadcastOp::jit_run " << @Root << " " << hccl_device_id << " " << xp << " " << yp;
    //ACLCHECK(aclrtSynchronizeStream(aclstream));
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
    HCCLCHECK(HcclBroadcast(@Root == hccl_device_id ? xp : yp, (uint64_t)x->num, hccl_dtype(x->dtype()), @Root, comm, aclstream));
    if (@Root == hccl_device_id) {
        ACLCHECK(aclrtMemcpy(yp, x->num * sizeof(Tx), xp, x->num * sizeof(Tx), ACL_MEMCPY_DEVICE_TO_DEVICE));
        ACLCHECK(aclrtSynchronizeDevice());
    }
    ACLCHECK(aclrtSynchronizeDevice());
    ACLCHECK(aclrtSynchronizeStream(aclstream));
}

#endif // JIT

} // jittor
