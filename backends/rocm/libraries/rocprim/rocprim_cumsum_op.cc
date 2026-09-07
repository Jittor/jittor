#include "rocprim_cumsum_op.h"
#include "scan.h"
#include "core/var.h"
#include "core/executor.h"
#include "mem/allocator.h"
#include "ops/op_register.h"
#include "runtime/backend.h"

namespace jittor {
namespace {
RocprimScanType scan_type(NanoString dtype) {
    if (dtype == ns_float32) return RocprimScanType::Float32;
    if (dtype == ns_float64) return RocprimScanType::Float64;
    if (dtype == ns_int32) return RocprimScanType::Int32;
    if (dtype == ns_int64) return RocprimScanType::Int64;
    throw std::invalid_argument("rocPRIM cumsum supports float32/float64/int32/int64");
}

void check_scan(hipError_t status) {
    USER_CHECK(status == hipSuccess) << "rocPRIM cumsum: " << hipGetErrorString(status);
}
} // namespace

RocprimCumsumOp::RocprimCumsumOp(Var* x, bool reverse) : x(x), reverse(reverse) {
    scan_type(x->dtype());
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    y = create_output(nullptr, x->dtype());
}

void RocprimCumsumOp::infer_shape() {
    USER_CHECK(x->shape.size() == 1 || x->shape.size() == 2)
        << "rocPRIM cumsum requires rank-1 or rank-2 input";
    y->set_shape(x->shape);
}

void RocprimCumsumOp::run() {
    if (!x->num) return;
    const size_t rows = x->shape.size() == 1 ? 1 : x->shape[0];
    const size_t count = x->shape[x->shape.size() - 1];
    const size_t row_bytes = count * x->dtype().dsize();
    auto stream = static_cast<hipStream_t>(
        backend_ops(BackendId::Rocm).compute_stream(y->device_id));
    const auto dtype = scan_type(x->dtype());
    size_t workspace_bytes = 0;
    check_scan(rocprim_scan(nullptr, workspace_bytes, x->mem_ptr, y->mem_ptr,
                            count, dtype, reverse, stream));
    // A non-null pointer distinguishes execution from rocPRIM's size query.
    Allocation workspace(runtime_executor().temp_allocator,
                         std::max(size_t(1), workspace_bytes));
    for (size_t row = 0; row < rows; ++row)
        check_scan(rocprim_scan(workspace.ptr, workspace_bytes,
            static_cast<const char*>(x->mem_ptr) + row * row_bytes,
            static_cast<char*>(y->mem_ptr) + row * row_bytes,
            count, dtype, reverse, stream));
}

VarPtr RocprimCumsumOp::grad(Var*, Var* dout, Var*, int) {
    auto scan = op_constructor<VarPtr, Var*, bool>("rocprim_cumsum");
    return scan(dout, !reverse);
}

} // namespace jittor
