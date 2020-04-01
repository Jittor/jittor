// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include "misc/cuda_flags.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "mem/allocator/cuda_host_allocator.h"
#include "mem/allocator/cuda_device_allocator.h"
#include "event_queue.h"

namespace jittor {

SFRLAllocator cuda_dual_host_allocator(&cuda_host_allocator, 0.3, 1<<28);
SFRLAllocator cuda_dual_device_allocator(&cuda_device_allocator, 0.3, 1<<28);
CudaDualAllocator cuda_dual_allocator;
DelayFree delay_free;

namespace cuda_dual_local {

list<Allocation> allocations;

static void free_caller() {
    allocations.pop_front();
}

}

void to_free_allocation(CUDA_HOST_FUNC_ARGS) {
    using namespace cuda_dual_local;
    event_queue.push(free_caller);
}

}

#endif
