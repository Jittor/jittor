// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <chrono>
#include <iomanip>
#include "common.h"

namespace jittor {

static inline int _lzcnt(int64 v) {
    #ifdef __clang__
    #if __has_feature(__builtin_ia32_lzcnt_u64)
        return __builtin_ia32_lzcnt_u64(v);
    #else
        return v ? __builtin_clzll(v) : 64;
    #endif
    #else
        return __builtin_clzll(v);
    #endif
}

struct SimpleProfiler {
    string name;
    int64 cnt;
    int64 total_ns;
    int64 sum;
    int64 pcnt[7] = {0};
    int64 pns[7] = {0};
    int64 last[7] = {0};

    void report() {
        std::cerr << "=============================\nSimpleProfiler [" << name << "] cnt: " << cnt 
            << " sum: " << sum << " speed: " << std::setprecision(3) << (sum*1.0/total_ns)
            << " total: " ;
        if (total_ns < 1.e3)
            std::cerr << total_ns << " ns" << std::endl;
        else if (total_ns < 1.e6)
            std::cerr << std::setprecision(3) << total_ns/1.e3 << " us" << std::endl;
        else if (total_ns < 1.e9)
            std::cerr << std::setprecision(3) << total_ns/1.e6 << " ms" << std::endl;
        else
            std::cerr << std::setprecision(3) << total_ns/1.e9 << " s" << std::endl;
        std::cerr << "          <32ns    <1us     <32us    <1ms     <32ms    <1s     >1s\n";
        std::cerr << "cnt: ";
        for (int i=0; i<7; i++) std::cerr << std::setw(9) << pcnt[i];
        std::cerr << "\n     ";
        for (int i=0; i<7; i++) std::cerr << std::setw(9) << std::setprecision(3) << pcnt[i]*1.0/cnt;
        std::cerr << "\ntime:";
        for (int i=0; i<7; i++) std::cerr << std::setw(9) << std::setprecision(3) << pns[i]*1.0/total_ns;
        std::cerr << "\nlast:";
        for (int i=0; i<7; i++) std::cerr << std::setw(9) << last[i];
        std::cerr << std::endl;
    }

    inline SimpleProfiler(string&& name): name(move(name)), cnt(0), total_ns(0), sum(0) {}
    inline ~SimpleProfiler() { report(); }
    inline void add(int64 time, int64 s) {
        auto nbit = 64 - _lzcnt(time);
        auto i = (nbit-1) / 5;
        if (i>6) i=6;
        cnt ++;
        sum += s;
        total_ns += time;
        pcnt[i] ++;
        pns[i] += time;
        last[i] = cnt;
    }
};

/*
example:
    {
        static SimpleProfiler _("array");
        SimpleProfilerGuard __(_);
        ......
    }
 */
struct SimpleProfilerGuard {
    SimpleProfiler* p;
    int64 s;
    std::chrono::high_resolution_clock::time_point start;
    inline SimpleProfilerGuard(SimpleProfiler& p, int64 s=1) : p(&p), s(s) {
        start = std::chrono::high_resolution_clock::now();
    }
    void finish() {
        this->~SimpleProfilerGuard();
        s = 0;
    }
    inline ~SimpleProfilerGuard() {
        if (!s) return;
        auto finish = std::chrono::high_resolution_clock::now();
        auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
        p->add(total_ns, s);
    }
};


DECLARE_FLAG(int, profiler_enable);

} // jittor