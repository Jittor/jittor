#include "memory_profiler.h"
#include "graph.h"
#include "var_holder.h"
#include "var.h"
#include "mem/allocator/sfrl_allocator.h"
#include <iomanip>
#include <algorithm>
#include <sys/sysinfo.h>
#include <sstream>

namespace jittor {

//TODO reuse from mem_info.cc
struct FloatOutput_ {
    double value;
    string scale;
    int base;
    string suffix;
    int p=4;
};
std::ostream& operator<<(std::ostream& os, const FloatOutput_& o) {
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
    std::pair<size_t, size_t> mem_info = get_memory_info();
    if (mem_info.first > max_used_memory_size) {
        max_used_memory_size = mem_info.first;

        allocations.clear();
        size_t memory_size = 0;
        vector<std::pair<string, size_t>> live_vars;
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
                    std::stringstream stream;
                    stream << var;
                    live_vars.push_back(std::make_pair(stream.str(), var->size));
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

bool MemoryProfiler::cmp(const std::pair<string, size_t>& a, const std::pair<string, size_t>& b) {
    return a.second > b.second;
}

void MemoryProfiler::display_max_memory_info() {
    Log log("", 'i', 0);
    std::sort(max_live_vars.begin(), max_live_vars.end(), cmp);
    log << "\n=====display_max_memory_info=====\n";
    log << "max used memory" << FloatOutput_{(double)max_used_memory_size, " KMG", 1024, "B"} << "\n";
    log << "max var memory" << FloatOutput_{(double)max_memory_size, " KMG", 1024, "B"} << "\n\n";
    log << "[Size]" << "[Percent]" << "[Var Info]" << "\n";
    for (int i = 0; i < max_live_vars.size(); ++i) {
        log << FloatOutput_{(double)max_live_vars[i].second, " KMG", 1024, "B"} << double(max_live_vars[i].second) / max_memory_size * 100 << "%" << max_live_vars[i].first << "\n\n";
    }
    log << "=========================\n";
    log.end();
}

void display_max_memory_info() {
    memory_profiler.display_max_memory_info();
}

} // jittor