// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "profiler/cache_info.h"
#include "op_compiler.h"
#include "misc/cstr.h"
#include "misc/nano_vector.h"

namespace jittor {

// @pyjt(profiler)
// @attrs(submodule)
struct Profiler {
    struct Info {
        uint64_t count;
        // time in us
        uint64_t time_max, time_min, time_total;
        // thoughtput in byte
        uint64_t in_total, out_total;
        // compute thoughtput in ops
        uint64_t compute_total;
        // cache test info
        unique_ptr<CacheInfo> cache_info;
        cstr stack_info;
        unordered_map<NanoVector, pair<uint64, uint64>> shapes;

        void update(int c, uint64_t t, uint64_t in, uint64_t out, uint64_t comp) {
            count += 1<<c;
            time_max = std::max(time_max, t>>c);
            time_min = std::min(time_min, t>>c);
            time_total += t;
            in_total += in<<c;
            out_total += out<<c;
            compute_total += comp<<c;
        }
    };
    // @pyjt(start)
    static void start(int64 warmup=0, int64 rerun=0);
    // @pyjt(stop)
    static void stop();
    /** report a table, first element is header of the table */
    // @pyjt(report)
    static vector<vector<string>> report(const string& sort_key="TotalTime");
    static vector<vector<string>> report_cache(const string& sort_key="TotalTime");
    
    static void record_and_run(
        jit_op_entry_t jit_entry,
        Op* op,
        const char* jit_key
    );
    
    int64_t warmup=0, rerun=0;
    unordered_map<string, Info> records;
    ~Profiler();
};

extern Profiler profiler;

DECLARE_FLAG(int, profiler_enable);

} // jittor