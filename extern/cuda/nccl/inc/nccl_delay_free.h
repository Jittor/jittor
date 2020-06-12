// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "misc/cuda_flags.h"
#include "event_queue.h"
#include "mem/allocator.h"
#include "nccl_warper.h"

namespace jittor {
constexpr int MAX_ALLOC = 65536;

struct NcclDelayFree;

extern NcclDelayFree nccl_delay_free;

void to_free_nccl_allocation(CUDA_HOST_FUNC_ARGS);

struct NcclDelayFreeAllocation {
    Allocator* allocator;
    size_t allocation;
    int64 ref_cnt = 0;
    NcclDelayFreeAllocation(Allocator* allocator = NULL, size_t allocation = 0, int64 ref_cnt = 0) : allocator(allocator), allocation(allocation), ref_cnt(ref_cnt) {}
};

struct NcclDelayFree final : Allocator {
    int n_free_id = MAX_ALLOC;
    int free_ids[MAX_ALLOC];
    NcclDelayFreeAllocation nccl_delay_free_allocations[MAX_ALLOC];
    
    list<Allocation> wait_for_free;
    
    NcclDelayFree() {
        for (int i=0; i<MAX_ALLOC; i++)
            free_ids[i] = i;
    }
    
    ~NcclDelayFree();
    

    inline uint64 flags() const override { return _cuda; };
    const char* name() const override { return "nccl_delay_free"; };
    
    void* alloc(size_t size, size_t& allocation) override { 
        LOGf << "Should not call this";
        return nullptr;
    }
    
    void registe(Allocator*& allocator, size_t& allocation) {
        // 如果已经是nccl delay free了，无事发生
        if (allocator == this)
            return;
        ASSERT(n_free_id);
        int id = free_ids[--n_free_id];
        nccl_delay_free_allocations[id] = NcclDelayFreeAllocation(allocator, allocation, 1);
        allocator = this;
        allocation = id;
    }
    
    bool share_with(size_t size, size_t allocation) override {
        nccl_delay_free_allocations[allocation].ref_cnt ++;
        return true;
    }
    
    void free(void* mem_ptr, size_t size, const size_t& allocation) override {
        auto& loc = nccl_delay_free_allocations[allocation];
        if (!--loc.ref_cnt) {
            free_ids[n_free_id++] = allocation;
            wait_for_free.emplace_back(mem_ptr, loc.allocation, size, loc.allocator);
            // 加一个 callback 
            checkCudaErrors(_cudaLaunchHostFunc(all_reduce_s, &to_free_nccl_allocation, 0));
            
        }
    }
};
}//jittor

#endif
