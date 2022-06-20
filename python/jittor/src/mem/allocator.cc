// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <typeinfo>
#include "misc/cuda_flags.h"
#include "var.h"

#include "mem/allocator/aligned_allocator.h"
#ifdef HAS_CUDA
#include "mem/allocator/cuda_managed_allocator.h"
#include "mem/allocator/cuda_device_allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#endif
#include "mem/allocator/opencl_device_allocator.h"
#include "mem/allocator/stat_allocator.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/nfef_allocator.h"
#include "mem/allocator/temp_allocator.h"

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

Allocator* get_allocator(bool temp_allocator) {
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
    if (use_opencl) {
        use_sfrl_allocator = 0;
        use_nfef_allocator = 1;
        LOGvv << "Using opencl_device_allocator";
        allocator = &opencl_device_allocator;
    }
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

#ifdef HAS_OPENCL

void migrate_opencl_to_cpu(Var* var, Allocator* allocator) {
    Allocation a(cpu_allocator, var->size);
    cl_int err;
    err=clEnqueueReadBuffer(opencl_queue, *(cl_mem*)var->mem_ptr, CL_TRUE, 0, var->size, a.ptr, 0, 0, 0);
    if(err != CL_SUCCESS)
        LOGf << "Can't read back data to host";
    var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = a.ptr;
    var->allocation = a.allocation;
    var->allocator = a.allocator;
    a.ptr = nullptr;
}


void migrate_cpu_to_opencl(Var* var, Allocator* allocator) {
    Allocation a(allocator, var->size);
    cl_int err;
    err=clEnqueueWriteBuffer(opencl_queue, *(cl_mem*)a.ptr, CL_TRUE, 0, var->size, var->mem_ptr, 0, 0, 0);
    if(err != CL_SUCCESS) 
        LOGf << "Can't write data to device";
    var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = a.ptr;
    var->allocation = a.allocation;
    var->allocator = a.allocator;
    a.ptr = nullptr;
}

#endif

#ifdef HAS_CUDA

void migrate_to_cpu(Var* var, Allocator* allocator) {
    if (!use_cuda_managed_allocator)
        allocator = cpu_allocator;
    if (var->allocator == &delay_free) {
        var->allocator = allocator;
        delay_free.migrate_to_cpu(
            var->mem_ptr, var->allocation, var->size, var->allocator
        );
    } else
    if (!use_cuda_managed_allocator) {
        // must be a device allocator
        Allocation a(allocator, var->size);
        checkCudaErrors(cudaMemcpy(a.ptr, var->mem_ptr, var->size, cudaMemcpyDeviceToHost));
        var->allocator->free(var->mem_ptr, var->size, var->allocation);
        var->mem_ptr = a.ptr;
        var->allocation = a.allocation;
        var->allocator = a.allocator;
        a.ptr = nullptr;
    }
}


void migrate_to_gpu(Var* var, Allocator* allocator) {
    // only happend when not using use_cuda_managed_allocator
    Allocation a(allocator, var->size);
    checkCudaErrors(cudaMemcpy(a.ptr, var->mem_ptr, var->size, cudaMemcpyHostToDevice));
    var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = a.ptr;
    var->allocation = a.allocation;
    var->allocator = a.allocator;
    a.ptr = nullptr;
}

#endif

} // jittor