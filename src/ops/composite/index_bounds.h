// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Jittor core maintainers.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstdio>
#include "core/common.h"

namespace jittor {

#if defined(JIT_cuda) && !defined(IS_ACL)
#define JT_INDEX_BOUNDS_HD __host__ __device__
#else
#define JT_INDEX_BOUNDS_HD
#endif

// Where an out-of-range index was seen, so the host can name it after the loop.
//
// The gather and scatter kernels cannot throw from inside their loop: on CPU it
// is an OpenMP region, and on CUDA it is a kernel. So the loop records and
// clamps, and the caller raises once the loop is over. Clamping is what keeps
// the read or write inside the buffer in the meantime -- without it the process
// would fault before reaching the check, which is the very failure being fixed.
//
// The writes race under OpenMP. That is deliberate and harmless: every writer
// stores to the same three fields, and any one of the offending indices is
// enough to name the fault. Ordering between them is not information anyone
// wants.
struct IndexFault {
    int64 index = 0;
    int64 size = 0;
    int dim = -1;  // -1 means nothing was recorded
    int claimed = 0;  // device only: the one thread allowed to write the record

    inline bool bad() const { return dim >= 0; }
};

// A record the device can write and the host can still read after the kernel
// has trapped: page-locked host memory mapped into the device's address space.
// Device memory is no use for this, since the trap takes the CUDA context with
// it and nothing on the device can be read back afterwards.
//
// Allocated on first use by the backend that registered an allocator; null
// when there is none, and the kernels then fall back to printing the index.
// The allocator stores the host and device addresses of one IndexFault.
using DeviceIndexFaultAllocator = bool (*)(IndexFault** host, IndexFault** device);
EXTERN_LIB void set_device_index_fault_allocator(DeviceIndexFaultAllocator allocator);
// The device address to hand to a kernel. Read back by
// `take_device_index_fault` (runtime/launch_diagnostics.h).
EXTERN_LIB IndexFault* device_index_fault();

// Normalise one index against the size of the dimension it addresses.
//
// A negative index counts from the end, which is Python's convention and is
// what the kernels have always done. What they did not do is check the result,
// so `x[jt.array([99])]` on a length-5 tensor read whatever followed the
// buffer: 0.0 for a small overshoot, a segfault for a large one. Python `int`
// indices were already checked at build time (`getitem_op.cc`), and slices
// clamp to the tensor the way NumPy does; an index arriving in a Var was the
// one path with no check at all, and it is the path embedding lookups take.
//
// On device there is nobody to report to, so the kernel records the offending
// index in `device_index_fault()` and traps. That aborts the launch and
// surfaces as a CUDA error at the next synchronisation, whose message then
// names the index (helper_cuda.h). It is the same bargain PyTorch makes for its
// device-side asserts: the context does not survive, and that is still better
// than a silent wrong answer.
//
// Only the first offending thread writes the record. `dim` goes last, behind a
// system fence, so the host never reads a half-written one. Every other thread
// prints and traps exactly as before: one of them may trap first, and a thread
// that merely returned would leave a record already claimed by an earlier
// fault on another device to silently clamp this one.
JT_INDEX_BOUNDS_HD inline int64 bounded_index(int64 v, int64 size, int dim,
                                              IndexFault* fault) {
    int64 w = v < 0 ? v + size : v;
    if (w < 0 || w >= size) {
#ifdef __CUDA_ARCH__
        if (fault && atomicCAS(&fault->claimed, 0, 1) == 0) {
            fault->index = v;
            fault->size = size;
            __threadfence_system();
            fault->dim = dim;
            __threadfence_system();
        } else {
            printf("[jittor] index %lld is out of bounds for dimension %d "
                   "with size %lld\n", (long long)v, dim, (long long)size);
        }
        __trap();
#else
        if (fault) {
            fault->index = v;
            fault->size = size;
            fault->dim = dim;
        }
#endif
        return 0;
    }
    return w;
}

} // jittor
