#include "rocm_arg_reduce_op.h"
#include "arg_reduce.h"
#include "core/var.h"
#include "ops/composite/arg_reduce_op.h"
#include "ops/composite/op_capability.h"
#include <climits>

namespace jittor {
namespace {
bool supported(Var* input, Var*, NanoString, bool) {
    return input->dtype() == ns_float32 || input->dtype() == ns_float64
        || input->dtype() == ns_int32 || input->dtype() == ns_int64;
}
RegisterOpCapability<vector<VarPtr>, Var*, Var*, NanoString, bool> capability(
    BackendId::Rocm, OpCapability::SegmentedArgReduce, "rocm_arg_reduce", supported,
    {"float32", "float64", "int32", "int64"});
} // namespace

RocmArgReduceOp::RocmArgReduceOp(Var* input, Var* offsets, NanoString operation, bool keepdims)
    : offsets(offsets), operation(operation), keepdims(keepdims) {
    USER_CHECK(supported(input, offsets, operation, keepdims)) << "Unsupported ROCm arg-reduction dtype";
    USER_CHECK(operation == ns_maximum || operation == ns_minimum) << "Expected argmax or argmin";
    USER_CHECK(input->num <= INT_MAX && offsets->dtype() == ns_int32)
        << "ROCm arg-reduction requires int32 segment offsets";
    auto contiguous = op_constructor<VarPtr, Var*>("contiguous");
    x = contiguous(input);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_manual_set_vnbb);
    indices = create_output(nullptr, ns_int32);
    values = create_output(nullptr, x->dtype());
    indices->set_flag(VarFlags::_needed_by_backward);
}

void RocmArgReduceOp::infer_shape() {
    USER_CHECK(x->shape.size() && x->shape[x->shape.size() - 1] > 0) << "Cannot arg-reduce an empty axis";
    NanoVector shape;
    int64 rows = 1;
    for (int i = 0; i + 1 < x->shape.size(); ++i) {
        shape.push_back(x->shape[i]);
        rows *= x->shape[i];
    }
    USER_CHECK(offsets->shape.size() == 1 && offsets->num == rows + 1)
        << "ROCm arg-reduction segment count does not match input";
    if (keepdims) shape.push_back(1);
    indices->set_shape(shape);
    values->set_shape(shape);
    if (!shape.size()) {
        indices->set_flag(VarFlags::_is_scalar);
        values->set_flag(VarFlags::_is_scalar);
    }
}

void RocmArgReduceOp::run() {
    const auto dtype = x->dtype() == ns_float32 ? RocprimScanType::Float32
        : x->dtype() == ns_float64 ? RocprimScanType::Float64
        : x->dtype() == ns_int32 ? RocprimScanType::Int32 : RocprimScanType::Int64;
    auto stream = static_cast<hipStream_t>(backend_ops(BackendId::Rocm).compute_stream(values->device_id));
    auto status = rocm_arg_reduce(x->mem_ptr, offsets->ptr<int>(), int(values->num),
        values->mem_ptr, indices->ptr<int>(), dtype, operation == ns_minimum, stream);
    CHECK(status == hipSuccess) << "ROCm arg-reduction: " << hipGetErrorString(status);
}

VarPtr RocmArgReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return ArgReduceOp::get_grad(out, dout, v, v_index, int(v->shape.size()) - 1, indices);
}
} // namespace jittor
