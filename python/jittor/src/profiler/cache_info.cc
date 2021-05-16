// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "profiler/cache_info.h"

namespace jittor {
CacheInfo::CacheInfo(unique_ptr<MemoryChecker>* mm) {
    check_times = mm->get()->check_times;
    tlb_miss_times = mm->get()->tlb->miss_time;
    cache_miss_times.clear();    
    for (int i = 0; i < (int)mm->get()->caches.size(); ++i)
        cache_miss_times.push_back(mm->get()->caches[i]->miss_time);
}

CacheInfo::CacheInfo() {
    check_times = tlb_miss_times = 0;
    cache_miss_times.clear();    
}

} //jittor