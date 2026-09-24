#pragma once

namespace jittor {

// Maximum team size for the calling thread's next OpenMP parallel region.
// This is a host runtime limit, not an accelerator worker count.
// @pyjt(runtime_openmp_max_threads)
int runtime_openmp_max_threads();

} // namespace jittor
