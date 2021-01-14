// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
#include "mem/allocator/temp_allocator.h"
#include "mem/mem_info.h"
#include "update_queue.h"
#include "executor.h"

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

void display_memory_info(const char* fileline, bool dump_var, bool red_color) {
    int p = 3;
    Log log(fileline, red_color?'e':'i', 0);
    log << "\n=== display_memory_info ===\n";
    log << "total_cpu_ram:" << 
        FloatOutput{(double)mem_info.total_cpu_ram, " KMG", 1024, "B"};
    log << "total_cuda_ram:" << 
        FloatOutput{(double)mem_info.total_cuda_ram, " KMG", 1024, "B"} >> "\n";
    log << "hold_vars:" << VarHolder::hold_vars.size()
        << "lived_vars:" << Var::number_of_lived_vars
        << "lived_ops:" << Op::number_of_lived_ops >> '\n';
    log << "update queue:" << update_queue.queue.size() 
        >> '/' >> update_queue.map.size() >> '\n';

    #ifdef NODE_MEMCHECK
    // get the oldest var
    // vector<Node*> queue;
    // auto t = ++Node::tflag_count;
    // for (auto& vh : VarHolder::hold_vars)
    //     if (vh->var->tflag != t) {
    //         vh->var->tflag = t;
    //         queue.push_back(vh->var);
    //     }
    // bfs_both(queue, [](Node*){return true;});
    // vector<pair<int64, Node*>> nodes;
    // nodes.reserve(queue.size());
    // for (auto* node : queue)
    //     nodes.push_back({node->__id(), node});
    // std::sort(nodes.begin(), nodes.end());
    // log << "list of the oldest nodes:\n";
    // for (int i=0; i<10 && i<nodes.size(); i++) {
    //     log << "ID#" >> nodes[i].first >> ":" << nodes[i].second << "\n";
    // }
    #endif

    if (use_stat_allocator) {
        log << "stat:" << use_stat_allocator;
        log << "total alloc:" << FloatOutput{(double)(stat_allocator_total_alloc_byte 
                        - stat_allocator_total_free_byte), " KMG", 1024, "B"};
        log << "total alloc call:" << FloatOutput{(double)(stat_allocator_total_alloc_call 
                        - stat_allocator_total_free_call), " KMG", 1000, ""}
            >> '(' >> stat_allocator_total_alloc_call >> '/' >> 
            stat_allocator_total_free_call >> ")\n";
    }
    int64 all_total = 0, gpu_total = 0, cpu_total = 0;
    for (auto& a : SFRLAllocator::sfrl_allocators) {
        auto total = a->used_memory + a->unused_memory;
        all_total += total;
        a->is_cuda() ? gpu_total += total : cpu_total += total;
        log << "name:" << a->name() << "is_cuda:" << a->is_cuda()
            << "used:" << FloatOutput{(double)a->used_memory, " KMG", 1024, "B"}
                >> "(" >> std::setprecision(p) >> a->used_memory*100.0 / total >> "%)"
            << "unused:" << FloatOutput{(double)a->unused_memory, " KMG", 1024, "B"} 
                >> "(" >> std::setprecision(p) >> a->unused_memory*100.0 / total >> "%)"
            << "total:" << FloatOutput{(double)total, " KMG", 1024, "B"} >> "\n";
    }
    log << "cpu&gpu:" << FloatOutput{(double)all_total, " KMG", 1024, "B"}
        << "gpu:" << FloatOutput{(double)gpu_total, " KMG", 1024, "B"}
        << "cpu:" << FloatOutput{(double)cpu_total, " KMG", 1024, "B"} >> '\n';
    if (use_temp_allocator) {
        TempAllocator* temp_allocator = (TempAllocator*)exe.temp_allocator;
        log << "\nname:" << temp_allocator->name() << "\n";
        log << "used_memory:" << FloatOutput{(double)temp_allocator->used_memory, " KMG", 1024, "B"} << "\n";
        log << "unused_memory:" << FloatOutput{(double)temp_allocator->unused_memory, " KMG", 1024, "B"} << "\n";

    }
    if (dump_var) {
        vector<Node*> queue;
        unordered_set<Node*> visited;
        for (auto& vh : VarHolder::hold_vars)
            if (!visited.count(vh->var)) {
                queue.push_back(vh->var);
                visited.insert(vh->var);
            }
        int64 cum = 0;
        for (int i=0; i<queue.size(); i++) {
            for (auto* n : queue[i]->inputs())
                if (!visited.count(n)) {
                    queue.push_back(n);
                    visited.insert(n);
                }
            for (auto* n : queue[i]->outputs())
                if (!visited.count(n)) {
                    queue.push_back(n);
                    visited.insert(n);
                }
            if (queue[i]->is_var()) {
                auto v = (Var*)queue[i];
                if (v->size>=0 && v->mem_ptr) {
                    cum += v->size;
                    log << FloatOutput{(double)v->size, " KMG", 1024, "B"}
                        >> "(" >> std::setprecision(p) >> v->size*100.0 / all_total >> "%)"
                        << FloatOutput{(double)cum, " KMG", 1024, "B"}
                        >> "(" >> std::setprecision(p) >> cum*100.0 / all_total >> "%)"
                        << v >> "\n";
                    if (v->size == 100*64*112*112*4) {
                        for (auto op : v->outputs())
                            log << "\t" << op << '\n';
                    }
                }
            }
        }
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