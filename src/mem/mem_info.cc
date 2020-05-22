// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <iomanip>
#include <algorithm>
#include <sys/sysinfo.h>

#include "var.h"
#include "op.h"
#include "var_holder.h"
#include "graph.h"
#include "misc/cuda_flags.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/stat_allocator.h"
#include "mem/mem_info.h"

namespace jittor {

struct FloatOutput {
    double value;
    string scale;
    int base;
    string suffix;
    int p=4;
};

std::ostream& operator<<(std::ostream& os, const FloatOutput& o) {
    int w = 8;
    os << std::setw(w-2-o.suffix.size());
    os << std::setprecision(o.p);
    uint i=0;
    double k = o.value;
    for (; i+1<o.scale.size(); i++) {
        if (k<o.base) break;
        k /= o.base;
    }
    os << k << o.scale[i];
    return os << o.suffix;
}

void display_memory_info(const char* fileline) {
    int p = 3;
    Log log(fileline, 'i', 0);
    log << "\n=== display_memory_info ===\n";
    log << "total_cpu_ram:" << 
        FloatOutput{(double)mem_info.total_cpu_ram, " KMG", 1024, "B"};
    log << "total_cuda_ram:" << 
        FloatOutput{(double)mem_info.total_cuda_ram, " KMG", 1024, "B"} >> "\n";
    log << "hold_vars:" << VarHolder::hold_vars.size()
        << "lived_vars:" << Var::number_of_lived_vars
        << "lived_ops:" << Op::number_of_lived_ops >> '\n';

    #ifdef NODE_MEMCHECK
    // get the oldest var
    vector<Node*> queue;
    auto t = ++Node::tflag_count;
    for (auto& vh : VarHolder::hold_vars)
        if (vh->var->tflag != t) {
            vh->var->tflag = t;
            queue.push_back(vh->var);
        }
    bfs_both(queue, [](Node*){return true;});
    vector<pair<int64, Node*>> nodes;
    nodes.reserve(queue.size());
    for (auto* node : queue)
        nodes.push_back({node->__id(), node});
    std::sort(nodes.begin(), nodes.end());
    log << "list of the oldest nodes:\n";
    for (int i=0; i<10 && i<nodes.size(); i++) {
        log << "ID#" >> nodes[i].first >> ":" << nodes[i].second << "\n";
    }
    #endif

    if (use_stat_allocator) {
        log << "stat:" << use_stat_allocator;
        log << "total alloc:" << FloatOutput{(double)(stat_allocator_total_alloc_byte 
                        - stat_allocator_total_free_byte), " KMG", 1024, "B"};
        log << "total alloc call:" << FloatOutput{(double)(stat_allocator_total_alloc_call 
                        - stat_allocator_total_free_call), " KMG", 1000, ""} >> '\n';
    }
    for (auto& a : SFRLAllocator::sfrl_allocators) {
        auto total = a->used_memory + a->unused_memory;
        log << "name:" << a->name() << "is_cuda:" << a->is_cuda()
            << "used:" << FloatOutput{(double)a->used_memory, " KMG", 1024, "B"}
                >> "(" >> std::setprecision(p) >> a->used_memory*100.0 / total >> "%)"
            << "unused:" << FloatOutput{(double)a->unused_memory, " KMG", 1024, "B"} 
                >> "(" >> std::setprecision(p) >> a->unused_memory*100.0 / total >> "%)"
            << "total:" << FloatOutput{(double)total, " KMG", 1024, "B"} >> "\n";
    }
    log >> "===========================\n";
    log.end();
}

MemInfo::MemInfo() {
    struct sysinfo info = {0};
    sysinfo(&info);
    total_cpu_ram = info.totalram;
    total_cuda_ram = 0;
#ifdef HAS_CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    total_cuda_ram = prop.totalGlobalMem;
#endif
}

MemInfo mem_info;

} // jittor