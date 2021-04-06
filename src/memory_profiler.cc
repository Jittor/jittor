// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "memory_profiler.h"
#include "graph.h"
#include "var_holder.h"
#include "var.h"
#include "mem/allocator/sfrl_allocator.h"
#include <iomanip>
#include <algorithm>
#include <sys/sysinfo.h>
#include <sstream>
#include "pybind/py_var_tracer.h"

namespace jittor {

//TODO reuse from mem_info.cc
struct FloatOutput_ {
    double value;
    string scale;
    int base;
    string suffix;
    int p=4;
};

inline std::ostream& operator<<(std::ostream& os, const FloatOutput_& o) {
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

MemoryProfiler memory_profiler;
DEFINE_FLAG(int, profile_memory_enable, 0, "Enable memory profiler.");

MemoryProfiler::MemoryProfiler() { 
    clear(); 
}

void MemoryProfiler::clear() {
    allocations.clear();
    max_memory_size = 0;
    max_used_memory_size = 0;
}

std::pair<size_t, size_t> MemoryProfiler::get_memory_info() {
    ASSERT(profile_memory_enable == 1);
    size_t used = 0;
    size_t unused = 0;
    //TODO add mssfrl allocator
    for (auto& a : SFRLAllocator::sfrl_allocators) {
        used += a->used_memory;
        unused += a->unused_memory;
    }
    return std::make_pair(used, unused);
}

void MemoryProfiler::check() {
    ASSERT(profile_memory_enable == 1);
    std::pair<size_t, size_t> mem_info = get_memory_info();
    if (mem_info.first > max_used_memory_size) {
        max_used_memory_size = mem_info.first;

        allocations.clear();
        size_t memory_size = 0;
        std::vector<std::pair<std::pair<string, vector<Stack>>, size_t>> live_vars;
        vector<Node*> queue;

        auto t = ++Node::tflag_count;
        for (auto& vh : VarHolder::hold_vars)
            if (vh->var->tflag != t) {
                vh->var->tflag = t;
                queue.push_back(vh->var);
            }
        bfs_both(queue, [](Node*){return true;});
        for (Node* node : queue) {
            if (node->is_var()) {
                Var* var = (Var*)node;
                if (var->mem_ptr != nullptr) {
                    vector<Stack> stacks = get_node_trace(var);
                    if (stacks.size() == 0) {
                        stacks.push_back(Stack());
                    }
                    std::stringstream stream;
                    stream << var;
                    live_vars.push_back(std::make_pair(std::make_pair(stream.str(), stacks), var->size));
                    if (!allocations.count(var->mem_ptr)) {
                        allocations[var->mem_ptr] = 1;
                        memory_size += var->size;
                    }
                }
            }
        }
        max_live_vars = live_vars;
        max_memory_size = memory_size;
    }
}

bool MemoryProfiler::cmp(const std::pair<std::pair<string, vector<Stack>>, size_t>& a, const std::pair<std::pair<string, vector<Stack>>, size_t>& b) {
    return a.second > b.second;
}

void MemoryProfiler::display_max_memory_info() {
    ASSERT(profile_memory_enable == 1);
    Log log("", 'i', 0);
    std::sort(max_live_vars.begin(), max_live_vars.end(), cmp);
    log << "\n=====display_max_memory_info=====\n";
    log << "max used memory" << FloatOutput_{(double)max_used_memory_size, " KMG", 1024, "B"} << "\n";
    log << "max var memory" << FloatOutput_{(double)max_memory_size, " KMG", 1024, "B"} << "\n\n";
    log << "[Size]" << "[Percent]" << "[Var Info]" << "\n";
    for (int i = 0; i < max_live_vars.size(); ++i) {
        log << FloatOutput_{(double)max_live_vars[i].second, " KMG", 1024, "B"} 
        << double(max_live_vars[i].second) / max_memory_size * 100 << "%" 
        << max_live_vars[i].first.first 
        << max_live_vars[i].first.second[0].file_path + ":" + std::to_string(max_live_vars[i].first.second[0].lineno)
        << "\n\n";
    }
    log << "=========================\n";
    log.end();
}

void display_max_memory_info() {
    ASSERT(profile_memory_enable == 1);
    memory_profiler.display_max_memory_info();
}

string MemoryProfiler::get_max_memory_info() {
    ASSERT(profile_memory_enable == 1);
    std::stringstream out;
    string div1 = "[!@#div1!@#]";
    string div2 = "[!@#div2!@#]";
    string div3 = "[!@#div3!@#]";

    std::sort(max_live_vars.begin(), max_live_vars.end(), cmp);
    out << max_memory_size;
    for (int i = 0; i < max_live_vars.size(); ++i) {
        out << div1;
        out << max_live_vars[i].first.first << div2;
        out << max_live_vars[i].second << div2;
        for (int j = 0; j < max_live_vars[i].first.second.size(); ++j) {
            out << max_live_vars[i].first.second[j].file_path + ":" + std::to_string(max_live_vars[i].first.second[j].lineno) << div3
                << max_live_vars[i].first.second[j].module_name << div3
                << max_live_vars[i].first.second[j].module_type << div2;
        }
    }
    return out.str();
}

string get_max_memory_info() {
    ASSERT(profile_memory_enable == 1);
    return memory_profiler.get_max_memory_info();
}

} // jittor