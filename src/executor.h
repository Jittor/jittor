// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "mem/allocator.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

namespace jittor {

struct Executor {
    Allocator* allocator;
    bool last_is_cuda = false;
    void run_sync(vector<Var*> vars, bool device_sync);
};

extern Executor exe;
    
} // jittor