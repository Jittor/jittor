// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <iomanip>
#include <algorithm>
#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/host_info.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#elif defined(_WIN32)
#include <windows.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#include "var.h"
#include "op.h"
#include "var_holder.h"
#include "graph.h"
#include "misc/cuda_flags.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/stat_allocator.h"
#include "mem/allocator/temp_allocator.h"
#include "mem/mem_info.h"
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
    log << "total_device_ram:" << 
        FloatOutput{(double)mem_info.total_cuda_ram, " KMG", 1024, "B"} >> "\n";
    log << "hold_vars:" << hold_vars.size()
        << "lived_vars:" << Var::number_of_lived_vars
        << "lived_ops:" << Op::number_of_lived_ops >> '\n';

    #ifdef NODE_MEMCHECK
    // get the oldest var
    // vector<Node*> queue;
    // auto t = ++Node::tflag_count;
    // for (auto& vh : hold_vars)
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
        log << "name:" << a->name() << "is_device:" << a->is_cuda()
            << "used:" << FloatOutput{(double)a->used_memory, " KMG", 1024, "B"}
                >> "(" >> std::setprecision(p) >> a->used_memory*100.0 / total >> "%)"
            << "unused:" << FloatOutput{(double)a->unused_memory, " KMG", 1024, "B"} 
                >> "(" >> std::setprecision(p) >> a->unused_memory*100.0 / total >> "%)"
            << "total:" << FloatOutput{(double)total, " KMG", 1024, "B"} >> "\n";
    }
    if (use_temp_allocator && exe.temp_allocator) {
        for (auto& a : TempAllocator::temp_allocators) {
            auto total = a->used_memory + a->unused_memory;
            all_total += total;
            a->is_cuda() ? gpu_total += total : cpu_total += total;
            log << "name:" << a->name() << "is_device:" << a->is_cuda()
                << "used:" << FloatOutput{(double)a->used_memory, " KMG", 1024, "B"}
                    >> "(" >> std::setprecision(p) >> a->used_memory*100.0 / total >> "%)"
                << "unused:" << FloatOutput{(double)a->unused_memory, " KMG", 1024, "B"} 
                    >> "(" >> std::setprecision(p) >> a->unused_memory*100.0 / total >> "%)"
                << "total:" << FloatOutput{(double)total, " KMG", 1024, "B"} >> "\n";
        }
    }
    log << "cpu&gpu:" << FloatOutput{(double)all_total, " KMG", 1024, "B"}
        << "gpu:" << FloatOutput{(double)gpu_total, " KMG", 1024, "B"}
        << "cpu:" << FloatOutput{(double)cpu_total, " KMG", 1024, "B"} >> '\n';
    
    size_t cpu_free = 0;
#if defined(__linux__)
    cpu_free = get_avphys_pages() * sysconf(_SC_PAGESIZE);
#elif defined(__APPLE__)
    {
        mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
        vm_statistics_data_t vmstat;
        if (KERN_SUCCESS == host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count)) {
            cpu_free = vmstat.free_count * sysconf(_SC_PAGESIZE);
        }
    }
#endif
    size_t gpu_free = 0, _gpu_total = 0;
    (void)gpu_free; (void)_gpu_total;
    #ifdef HAS_CUDA
    cudaMemGetInfo(&gpu_free, &_gpu_total);
    #endif
    log << "free: cpu(">>FloatOutput{(double)cpu_free, " KMG", 1024, "B"}
        >> ") gpu(">>FloatOutput{(double)gpu_free, " KMG", 1024, "B"} >> ")\n";
    if (dump_var) {
        vector<Node*> queue;
        unordered_set<Node*> visited;
        for (auto& vh : hold_vars)
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

    if (red_color) {
        bool gpu_overflow = (double)gpu_total>(double)mem_info.total_cuda_ram*0.95;
        bool cpu_overflow = (double)cpu_total>(double)mem_info.total_cpu_ram*0.95;
        // cpu total too small, not possible
        if (mem_info.total_cpu_ram < 100000)
            cpu_overflow = false;
        if(gpu_overflow || cpu_overflow) {
            double used = gpu_overflow ? (double)gpu_total : (double)cpu_total;
            double total = gpu_overflow ? (double)mem_info.total_cuda_ram : (double)mem_info.total_cpu_ram;
            log.end();
            LOGf << "\n*******************\n"
                >> (gpu_overflow?"GPU":"CPU") << "memory is overflow, please reduce your batch_size or data size!\nTotal:" << FloatOutput{(double)total, " KMG", 1024, "B"} << "Used:" << FloatOutput{(double)used, " KMG", 1024, "B"};
        } else
            return;
    }

    log.end();
}

EXTERN_LIB vector<void(*)()> sigquit_callback;

void meminfo_callback() {
    display_memory_info();
}

MemInfo::MemInfo() {

#if defined(__linux__)
    struct sysinfo info = {0};
    sysinfo(&info);
    total_cpu_ram = info.totalram;
#elif defined(__APPLE__)
    int mib[] = {CTL_HW, HW_MEMSIZE};
    size_t len=sizeof(total_cpu_ram);
    sysctl(mib, 2, &total_cpu_ram, &len, NULL, 0);
#elif defined(_WIN32)
    MEMORYSTATUSEX statex;
    GlobalMemoryStatusEx (&statex);
    total_cpu_ram = statex.ullTotalPhys;
#endif

    total_cuda_ram = 0;
#ifdef HAS_CUDA
    size_t gpu_free = 0, _gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &_gpu_total);
    total_cuda_ram = _gpu_total;
#endif
    sigquit_callback.push_back(&meminfo_callback);
}

MemInfo mem_info;

} // jittor