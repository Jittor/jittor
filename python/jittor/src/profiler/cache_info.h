// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <vector>
#include <memory>
#include "profiler/memory_checker.h"

namespace jittor {
struct CacheInfo {
    int64_t check_times, tlb_miss_times;
    vector<int64> cache_miss_times;
    CacheInfo(unique_ptr<MemoryChecker>* mm);
    CacheInfo();
};

} // jittor