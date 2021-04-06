// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cnrt.h>
#include "mlu_warper.h"
#include "mem/allocator.h"
#include "misc/mlu_flags.h"

namespace jittor {

// DECLARE_FLAG(int, use_mlu);

struct MLUDeviceAllocator : Allocator {
    uint64 flags() const override { return _mlu; }
    const char* name() const override;
    void* alloc(size_t size, size_t& allocation) override;
    void free(void* mem_ptr, size_t size, const size_t& allocation) override;
};

extern MLUDeviceAllocator mlu_device_allocator;

}

