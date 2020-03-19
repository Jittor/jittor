// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "profiler/cache_info.h"
#include "profiler/profiler.h"
#include "op_compiler.h"

namespace jittor {

struct ProfilerGuard {
    const char* key;
    bool alive;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;

    inline void start(int64 warmup=0, int64 rerun=0) {
        alive = true;
        start_time = std::chrono::high_resolution_clock::now();
    }

    inline void stop() {
        if (!alive) return;
        alive = false;
        stop_time = std::chrono::high_resolution_clock::now();
        
        auto iter = profiler.records.find(key);
        if (iter == profiler.records.end()) {
            profiler.records[key] = Profiler::Info{
                0, 0, -1ull, 0, 
                0, 0, 0
            };
            iter = profiler.records.find(key);
        }

        auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
        // 24ns function call overhead
        total_ns = std::max((int64_t)1, total_ns-24);
        iter->second.update(1, total_ns, 0, 0, 0);
    }
    
    inline ProfilerGuard(const char* _key) {
        key = _key;
        if (profiler_enable) {
            ProfilerGuard::start();
        }
    }

    inline ~ProfilerGuard() {
        if (profiler_enable) {
            ProfilerGuard::stop();
        }
    }
};

DECLARE_FLAG(int, profiler_enable);

} // jittor