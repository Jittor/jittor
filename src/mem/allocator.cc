// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <typeinfo>
#include <iomanip>

#include "var.h"
#include "op.h"
#include "var_holder.h"
#include "misc/cuda_flags.h"

#include "mem/allocator/aligned_allocator.h"
#ifdef HAS_CUDA
#include "mem/allocator/cuda_managed_allocator.h"
#include "mem/allocator/cuda_device_allocator.h"
#endif
#include "mem/allocator/stat_allocator.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/nfef_allocator.h"

namespace jittor {


struct pair_hash {
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2> &pair) const {
		return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
	}
};


std::unordered_map<
    pair<string, Allocator*>, 
    unique_ptr<Allocator>,
    pair_hash> allocators;

template <class T>
Allocator* setup_allocator(Allocator* underlying) {
    pair<string, Allocator*> key{typeid(T).name(), underlying};
    auto iter = allocators.find(key);
    if (iter != allocators.end()) return iter->second.get();
    auto a = std::make_unique<T>();
    auto* p = a.get();
    a->setup(underlying);
    allocators[key] = move(a);
    return p;
}

Allocator* cpu_allocator = setup_allocator<SFRLAllocator>(&aligned_allocator);

Allocator* get_allocator() {
    Allocator* allocator = nullptr;
#ifdef HAS_CUDA
    if (use_cuda && !allocator) {
        if (use_cuda_managed_allocator) {
            LOGvv << "Using cuda_managed_allocator";
            allocator = &cuda_managed_allocator;
        } else {
            LOGvv << "Using cuda_device_allocator";
            allocator = &cuda_device_allocator;
        }
    }
#endif
    if (!allocator) {
        LOGvv << "Using aligned_allocator";
        allocator = &aligned_allocator;
    }
    if (use_stat_allocator==1) {
        LOGvv << "Using stat_allocator";
        allocator = setup_allocator<StatAllocator>(allocator);
    }
    if (use_nfef_allocator) {
        LOGvv << "Using use_nfef_allocator";
        allocator = setup_allocator<NFEFAllocator>(allocator);
        return allocator;
    }
    if (use_sfrl_allocator) {
        LOGvv << "Using sfrl_allocator";
        allocator = setup_allocator<SFRLAllocator>(allocator);
    }
    if (use_stat_allocator==2) {
        LOGvv << "Using stat_allocator at last";
        allocator = setup_allocator<StatAllocator>(allocator);
    }
    return allocator;
}

void gc_all() {
    for (auto& kv : allocators) kv.second->gc();
}

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
    int p = 2;
    Log log(fileline, 'i', 0);
    log << "\n=== display_memory_info ===\n";
    log << "hold_vars:" << VarHolder::hold_vars.size()
        << "lived_vars:" << Var::number_of_lived_vars
        << "lived_ops:" << Op::number_of_lived_ops >> '\n';
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


} // jittor