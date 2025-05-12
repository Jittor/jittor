// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include <list>
#include <mutex>
#include <cstring>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "misc/cuda_flags.h"
#include "var.h"
#include "mem/allocator.h"
#include "mem/allocator/sfrl_allocator.h"

namespace jittor {

struct DualAllocation {
    size_t ref_cnt;
    void* host_ptr, * device_ptr;
    size_t host_allocation, device_allocation;
};

EXTERN_LIB SFRLAllocator cuda_dual_host_allocator;
EXTERN_LIB SFRLAllocator cuda_dual_device_allocator;
EXTERN_LIB bool no_cuda_error_when_free;

struct CudaDualAllocator : Allocator {
    //for recycle block_id
    static const size_t ID_LIMIT = 1 << 16;
    int n_free_ids;
    int free_ids[ID_LIMIT];
    DualAllocation allocations[ID_LIMIT];

    CudaDualAllocator() {
        n_free_ids = ID_LIMIT;
        for (int i=0; i<n_free_ids; i++)
            free_ids[i] = i;
    }

    uint64 flags() const override { return _cuda; }
    const char* name() const override { return "dual"; };
    void* alloc(size_t size, size_t& allocation) override {
        ASSERT(n_free_ids) << "id pool empty";
        n_free_ids--;
        allocation = free_ids[n_free_ids];
        auto& da = allocations[allocation];
        da.ref_cnt = 1;
        da.host_ptr = cuda_dual_host_allocator.alloc(size, da.host_allocation);
        da.device_ptr = cuda_dual_device_allocator.alloc(size, da.device_allocation);
        return da.device_ptr;
    }
    void free(void* mem_ptr, size_t size, const size_t& allocation) override {
        auto& da = allocations[allocation];
        da.ref_cnt--;
        if (!da.ref_cnt) {
            cuda_dual_host_allocator.free(da.host_ptr, size, da.host_allocation);
            cuda_dual_device_allocator.free(da.device_ptr, size, da.device_allocation);
            free_ids[n_free_ids++] = allocation;
        }
    }

    bool share_with(size_t size, size_t allocation) override {
        auto& da = allocations[allocation];
        da.ref_cnt++;
        return true;
    };

    inline DualAllocation get_dual_allocation(const size_t& allocation) {
        return allocations[allocation];
    }
};

EXTERN_LIB CudaDualAllocator cuda_dual_allocator;

namespace cuda_dual_local {
    
EXTERN_LIB list<Allocation> allocations;

}

void to_free_allocation(CUDA_HOST_FUNC_ARGS);

struct DelayFree final : Allocator {
    inline uint64 flags() const override { return _cuda; };
    const char* name() const override { return "delay_free"; };
    void* alloc(size_t size, size_t& allocation) override { 
        LOGf << "Should not call this";
        return nullptr;
    }
    bool share_with(size_t size, size_t allocation) override {
        return cuda_dual_allocator.share_with(size, allocation);
    };
    void free(void* mem_ptr, size_t size, const size_t& allocation) override {
        using namespace cuda_dual_local;
        if (no_cuda_error_when_free) return;
        allocations.emplace_back(mem_ptr, allocation, size, &cuda_dual_allocator);
        peekCudaErrors(_cudaLaunchHostFunc(0, &to_free_allocation, 0));
    }

    void migrate_to_cpu(void*& mem_ptr, size_t& allocation, size_t size, Allocator* allocator) {
        auto da = cuda_dual_allocator.get_dual_allocation(allocation);
        auto pre_allocation = allocation;
        auto offset = (int64)mem_ptr - (int64)da.device_ptr;

        mem_ptr = allocator->alloc(size, allocation);

        checkCudaErrors(cudaMemcpy(mem_ptr, 
            (void*)((int64)da.device_ptr+offset), size, cudaMemcpyDeviceToHost));
        // std::memcpy(mem_ptr, (void*)((int64)da.host_ptr+offset), size);
        free(da.device_ptr, size, pre_allocation);
    }
};

EXTERN_LIB DelayFree delay_free;

}

#endif
