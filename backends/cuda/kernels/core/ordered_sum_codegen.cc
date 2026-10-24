#include <stdint.h>
#include "core/var.h"
#include "core/op.h"
#include "ops/composite/op_capability.h"
#include "runtime/device.h"

// Register metadata only; importing the core must not initialize CUDA.
#if defined(IS_CUDA) && IS_CUDA
namespace jittor {
namespace {
bool supports_ordered_sum(Var* x, uint mask, uint keep) {
    if (construction_target_backend(x) != BackendId::Cuda || x->dtype() != ns_float32)
        return false;
    const auto rank = x->shape.size();
    if (rank < 1 || rank >= 32 || (mask >> rank) || (keep & ~mask)) return false;
    if (mask & (1u << (rank-1))) {
        // Scalar leaves through width128, aligned float4 leaves above it.
        // Widths >=32 retain complete warp rows; smaller widths need a
        // distinct subwarp tree and deliberately stay on the generic route.
        const auto columns = x->shape[rank-1];
        if (columns < 32 || columns > 4096) return false;
        // The wide float4 tree is validated for dense storage only. Preserve
        // the original strided ReduceOp route before OrderedSum can copy x.
        if (columns > 128 && !x->is_contiguous()) return false;
        for (uint i=0; i<rank; ++i) {
            if (x->shape[i] <= 0) return false;
            if (i < rank-1 && (mask >> i & 1) && x->shape[i] != 1) return false;
        }
        return x->num > 0 && x->num <= 0x7fffffffLL / 4;
    }
    // One separately validated ATen2.8 output-vector4 / block(32,4) tree.
    // Full leading-prefix only: coalescing [2,38,C] must reduce R=76, not 38.
    // Check BEFORE OrderedSum's default storage adaptation can copy a view.
    const uint leading_mask = (1u << (rank-1))-1;
    if (rank >= 2 && mask == leading_mask && x->shape[rank-1] == 1024 &&
        x->num == 76LL*1024) {
        for (uint i=0; i<rank; ++i)
            if (x->shape[i] <= 0) return false;
        // Offset views, even aligned ones, deliberately remain unvalidated.
        // Pending ordinary CUDA allocations are aligned; verify the eventual
        // actual pointer again in jit_run before any vector4 access.
        return x->is_contiguous() && x->storage_offset_bytes == 0 &&
            (!x->mem_ptr || !(reinterpret_cast<uintptr_t>(x->mem_ptr) & 15u));
    }
    if (rank < 2) return false;
    const uint penultimate = 1u << (rank-2);
    if (!(mask & penultimate) || (mask & (1u << (rank-1)))) return false;
    for (uint i=0; i<rank; ++i) {
        if (x->shape[i] <= 0) return false;
        if (i < rank-2 && (mask >> i & 1) && x->shape[i] != 1) return false;
    }
    const auto rows = x->shape[rank-2], columns = x->shape[rank-1];
    // Under these bounds ATen's output-vectorized path does not split the
    // reduction across warps. Singleton output columns are a different tree.
    if (rows >= 128 || columns > 64 || (columns == 1 && rows != 1)) return false;
    return x->num > 0 && x->num <= 0x7fffffffLL / 4;
}
RegisterOpCapability<VarPtr, Var*, uint, uint> ordered_sum(
    BackendId::Cuda, OpCapability::OrderedSum, "ordered_sum", supports_ordered_sum,
    {"float32"});
}
} // namespace jittor
#endif
