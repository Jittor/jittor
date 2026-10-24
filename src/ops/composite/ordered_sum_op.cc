#include "ops/composite/ordered_sum_op.h"
#include "ops/composite/op_capability.h"
#include "core/var.h"
#include "runtime/device.h"

namespace jittor {
#ifndef JIT
static auto make_broadcast = op_constructor<VarPtr, Var*, Var*, uint, uint>("broadcast_to");

static bool ordered_sum_supported(Var* x, uint mask, uint keep) {
    if (!x) return false;
    keep &= mask;
    return find_op_capability<VarPtr, Var*, uint, uint>(
        construction_target_backend(x), OpCapability::OrderedSum, x, mask, keep) != nullptr;
}

OrderedSumOp::OrderedSumOp(Var* x, uint mask, uint keep)
    : x(x), reduce_mask(mask), keepdims_mask(keep) {
    USER_CHECK(ordered_sum_supported(x, mask, keep))
        << "No native ordered-sum implementation for this device, dtype or geometry";
    // Default storage adaptation materializes a strided input on the same
    // device before construction. The CUDA kernel only receives dense data.
    USER_CHECK(x->is_contiguous()) << "ordered_sum requires adapted dense input";
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
}

void OrderedSumOp::infer_shape() {
    NanoVector shape;
    shape.clear();
    for (uint i=0; i<x->shape.size(); ++i) {
        if (!(reduce_mask >> i & 1)) shape.push_back(x->shape[i]);
        else if (keepdims_mask >> i & 1) shape.push_back(1);
    }
    y->set_shape(shape);
}

VarPtr OrderedSumOp::grad(Var*, Var* dout, Var* v, int v_index) {
    if (v_index != 0) return nullptr;
    return make_broadcast(dout, v, reduce_mask, keepdims_mask);
}

void OrderedSumOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype();
    jk << "«LAST_AXIS:" << int((reduce_mask >> (x->shape.size()-1)) & 1);
}
#else
#ifndef JIT_cuda
void OrderedSumOp::jit_run() {
    USER_CHECK(false) << "ordered_sum has no implementation for this backend";
}
#endif
#endif
} // namespace jittor
