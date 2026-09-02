// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <typeinfo>
#include "misc/cuda_flags.h"

#include "mem/allocator/aligned_allocator.h"
#ifdef HAS_CUDA
#include "mem/allocator/cuda_managed_allocator.h"
#include "mem/allocator/cuda_device_allocator.h"
#include "mem/allocator/cuda_host_allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#endif
#include "mem/allocator/stat_allocator.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/nfef_allocator.h"
#include "mem/allocator/temp_allocator.h"
#include "mem/swap.h"
#include "var.h"

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

DEFINE_FLAG_WITH_SETTER(int, use_cuda_host_allocator, 1, "use cuda host allocator for cpu memory globally");

void setter_use_cuda_host_allocator(int value) {
    #ifdef HAS_CUDA
    auto use_cuda_bk = use_cuda;
    use_cuda = 0;
    use_cuda_host_allocator = value;
    cpu_allocator = get_allocator();
    use_cuda = use_cuda_bk;
    #endif
}

extern int64 sfrl_large_block_size_device;

#ifdef HAS_CUDA
// One device-memory pool per CUDA device. The global instances stay the
// pools of device 0 so code that names them keeps working.
static vector<unique_ptr<CudaDeviceAllocator>> device_allocators;
static vector<unique_ptr<CudaManagedAllocator>> managed_allocators;

static Allocator* cuda_base_allocator(int device) {
    if (use_cuda_managed_allocator) {
        if (device == 0) return &cuda_managed_allocator;
        if ((int)managed_allocators.size() <= device) managed_allocators.resize(device+1);
        auto& a = managed_allocators[device];
        if (!a) { a = std::make_unique<CudaManagedAllocator>(); a->device_id = device; }
        return a.get();
    }
    if (device == 0) return &cuda_device_allocator;
    if ((int)device_allocators.size() <= device) device_allocators.resize(device+1);
    auto& a = device_allocators[device];
    if (!a) { a = std::make_unique<CudaDeviceAllocator>(); a->device_id = device; }
    return a.get();
}
#endif

Allocator* get_allocator(bool temp_allocator) {
    int device = -1;
#ifdef HAS_CUDA
    if (use_cuda) device = current_device();
#endif
    return get_allocator(device, temp_allocator);
}

Allocator* get_allocator(int device, bool temp_allocator) {
    Allocator* allocator = nullptr;
    if (use_cuda && sfrl_large_block_size_device >= (1ll<<40)) {
        // if super large block is used, don't use
        // temp allocator
        temp_allocator = false;
    }
#ifdef HAS_CUDA
    if (use_cuda && device >= 0 && !allocator) {
        LOGvv << "Using cuda allocator of device" << device;
        allocator = cuda_base_allocator(device);
    } else
    if (use_cuda_host_allocator) {
        // The cuda host allocator (pinned memory via cudaMallocHost) requires a
        // real CUDA device; with none visible (e.g. CUDA_VISIBLE_DEVICES="" in a
        // CPU-only Ray actor) it aborts with cudaErrorNoDevice. The flag defaults
        // to 1 even in a .so built with HAS_CUDA, and the import-time reset isn't
        // always reliable, so gate on the actual device count here and fall back
        // to the plain aligned CPU allocator when there is no device -- keeping
        // jittor usable as a pure-CPU runtime. Cached: device visibility is fixed
        // for a process's lifetime.
        static int _cuda_dev_cnt = get_device_count();
        if (_cuda_dev_cnt > 0)
            allocator = &cuda_host_allocator;
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
    if (temp_allocator && use_temp_allocator) {
        LOGvv << "Using temp_allocator";
        allocator = setup_allocator<TempAllocator>(allocator);
    } else if (use_sfrl_allocator) {
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

static void migrate_empty_var(Var* var, Allocator* allocator) {
    Allocation target(allocator, 0);
    if (var->mem_ptr && var->allocator)
        var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = target.ptr;
    var->allocation = target.allocation;
    var->allocator = target.allocator;
    target.ptr = nullptr;
}

void migrate_to_cpu(Var* var, Allocator* allocator) {
    #ifdef HAS_CUDA
    if (!use_cuda_managed_allocator)
        allocator = cpu_allocator;
    #endif
    if (var->size == 0) {
        migrate_empty_var(var, allocator);
        return;
    }
    if (save_mem) {
        if (swap_timestamp != var->tflag) {
            swap_timestamp = ++tflag_count;
            var->tflag = swap_timestamp;
        }
        move_with_swap(var, cpu_allocator, true);
        return;
    }
    #ifdef HAS_CUDA
    if (var->allocator == &delay_free) {
        var->allocator = allocator;
        delay_free.migrate_to_cpu(
            var->mem_ptr, var->allocation, var->size, var->allocator
        );
    } else
    if (!use_cuda_managed_allocator) {
        if (!var->allocator->is_cuda()) return;
        // must be a device allocator. Issue the copy from the Var's own
        // device so it is ordered after the kernel that produced it.
        Allocation a(allocator, var->size);
        int dev = var->allocator->device(), prev = current_device();
        if (dev >= 0 && dev != prev) set_current_device(dev);
        checkCudaErrors(cudaMemcpy(a.ptr, var->mem_ptr, var->size, cudaMemcpyDeviceToHost));
        if (dev >= 0 && dev != prev) set_current_device(prev);
        var->allocator->free(var->mem_ptr, var->size, var->allocation);
        var->mem_ptr = a.ptr;
        var->allocation = a.allocation;
        var->allocator = a.allocator;
        a.ptr = nullptr;
    }
    #endif
}


void migrate_to_gpu(Var* var, Allocator* allocator) {
    #ifdef HAS_CUDA
    // only happend when not using use_cuda_managed_allocator
    if (var->size == 0) {
        migrate_empty_var(var, allocator);
        return;
    }
    if (save_mem) {
        if (swap_timestamp != var->tflag) {
            swap_timestamp = ++tflag_count;
            var->tflag = swap_timestamp;
        }
        move_with_swap(var, allocator, true);
        return;
    }
    Allocation a(allocator, var->size);
    checkCudaErrors(cudaMemcpy(a.ptr, var->mem_ptr, var->size, cudaMemcpyHostToDevice));
    var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = a.ptr;
    var->allocation = a.allocation;
    var->allocator = a.allocator;
    a.ptr = nullptr;
    #endif
}

} // jittor
