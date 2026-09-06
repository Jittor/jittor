#include "runtime/backend.h"

namespace jittor {

BackendOps make_corex_backend() {
    auto ops = make_cuda_backend();
    ops.id = BackendId::Corex;
    ops.name = "corex";
    ops.execution.prefer_compaction_kernel = true;
    ops.execution.supports_generated_device_kernels = true;
    ops.execution.warp_shuffle_width = 64;
    ops.execution.ordered_float_atomics = true;
    return ops;
}

} // namespace jittor
