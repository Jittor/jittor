// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opencl_warper.h"
#ifdef HAS_OPENCL
#include "mem/allocator.h"
#include "misc/opencl_flags.h"

namespace jittor {

struct OpenclDeviceAllocator : Allocator {
    uint64 flags() const override { return _opencl; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

extern OpenclDeviceAllocator opencl_device_allocator;

}

#endif