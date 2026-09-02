// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif
#include <stdio.h>
#include <memory>
#include <random>
#include <thread>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "var.h"
#include "mem/swap.h"
#include "mem/mem_info.h"

namespace jittor {

int64 swap_timestamp;
int64 swap_total;
constexpr int64 SWAP_BUF_SIZE = 1<<23; // 8M
extern string cache_path;

// Swap file names used to be built from a pid captured at static-init time, so
// a forked child kept writing the *parent's* file names -- and var ids collide
// across a fork too, because total_node is inherited. Read the pid at call
// time and mix in a per-process random token: the token also keeps a stale file
// left by a dead process with a recycled pid from being picked up.
static string swap_file_prefix() {
    static int cached_pid = -1;
    static string prefix;
    int pid = getpid();
    if (pid != cached_pid) {
        std::random_device rd;
        uint64 token = ((uint64)rd() << 32) | rd();
        cached_pid = pid;
        prefix = cache_path + "/tmp/swap-" + S(pid) + "-" + S(token) + "-";
    }
    return prefix;
}

static string swap_file_path(Var* x) {
    return swap_file_prefix() + S(x->id) + ".bin";
}

DEFINE_FLAG(int64, cpu_mem_limit, -1, "cpu_mem_limit");
DEFINE_FLAG(int64, device_mem_limit, -1, "device_mem_limit");

struct Swap {
    map<pair<int64,int64>, Var*> lived;
};

unordered_map<Allocator*, Swap> swaps;

void swap_to_disk(Var* x, Swap& swap) {
    swap_total += x->size;
    ASSERT(!x->flag(VarFlags::_is_swapped));
    // "search share_with" on swap.h's TODO list: a var that is a sub-range of
    // another var's block cannot be swapped out on its own -- the block stays
    // occupied and the other alias would read a hole. Refuse loudly rather
    // than corrupt it silently.
    ASSERT(!x->share_next) << "cannot swap out an aliased var" << x;
    string path = swap_file_path(x);
    #ifdef HAS_CUDA
    if (x->allocator->is_cuda()) {
        // was a function-local `static char* buffer = new char[8MB]`: leaked,
        // and two threads swapping at once trampled each other's staging area
        int64 buf_size = std::min(x->size, SWAP_BUF_SIZE);
        std::unique_ptr<char[]> buf(new char[buf_size]);
        char* buffer = buf.get();
        auto* memptr = (char*)x->mem_ptr;
        auto* fd = fopen(path.c_str(), "wb");
        CHECK(fd) << "swap file open failed:" << path << x;
        for (int64 i=0; i<x->size; i+=SWAP_BUF_SIZE) {
            int64 cp_size = std::min(x->size-i, SWAP_BUF_SIZE);
            // the return value used to be dropped: a failed D2H copy wrote the
            // staging buffer's previous contents to disk as if it were the var
            checkCudaErrors(cudaMemcpy(buffer, memptr+i, cp_size, cudaMemcpyDeviceToHost));
            auto res = fwrite(buffer, cp_size, 1, fd);
            if (res!=1) {
                fclose(fd);
                LOGf << "swap file write failed" << path << x;
            }
        }
        fclose(fd); 
    } else
    #endif
    {
        auto* fd = fopen(path.c_str(), "wb");
        CHECK(fd) << "swap file open failed:" << path << x;
        auto res = fwrite(x->mem_ptr, x->size, 1, fd);
        CHECK(res==1) << "failed to write swap file" << path << res << x->size << x;
        fclose(fd); 
    }
    auto iter = swap.lived.find({x->size, x->id});
    ASSERT(iter != swap.lived.end());
    swap.lived.erase(iter);
    x->allocator->free(x->mem_ptr, x->size, x->allocation);
    x->mem_ptr = nullptr;
    x->allocator = nullptr;
    x->allocation = 0;
    x->set_flag(VarFlags::_is_swapped);
}

bool alloc_with_swap(Var* x, Allocator* allocator, bool force) {

    auto& swap = swaps[allocator];
    // "allocator is set" used to mean either of two things: already allocated,
    // or a pending share_with whose Var* was parked in that field. Both skip
    // the limit/swap machinery below, so both are spelled out here.
    if (x->mem_ptr || x->is_sharing()) {
        // shared memory, no need alloc
        if (x->alloc(allocator)) {
            swap.lived[{x->size, x->id}] = x;
            return true;
        }
    }
    bool is_cpu = !allocator->is_cuda();
    int64 limit = is_cpu ? cpu_mem_limit : device_mem_limit;
    if (limit < 0) limit = 1ll<<60;
    if (allocator->used_memory + allocator->unused_memory + x->size > limit)
        allocator->gc();
    if (force && allocator->used_memory + allocator->unused_memory + x->size > limit) {
        auto iter = swap.lived.upper_bound({x->size, -1});
        auto unused_target = allocator->unused_memory + x->size;
        while (iter != swap.lived.end()) {
            auto* var = iter->second;
            iter++;
            if (var->tflag == swap_timestamp)
                continue;
            ASSERT(var->mem_ptr) << var->exist() << (void*)var << iter->first << (display_memory_info(), 1);
            if (!is_cpu) {
                // try move to cpu
                if (!move_with_swap(var, cpu_allocator, false))
                    swap_to_disk(var, swap);
            } else
                swap_to_disk(var, swap);
            if (allocator->used_memory + allocator->unused_memory + x->size <= limit || allocator->unused_memory >= unused_target) break;
        }
        // if still no space, swap other smaller var
        if (!(allocator->used_memory + allocator->unused_memory + x->size <= limit || allocator->unused_memory >= unused_target)) {
            auto iter = swap.lived.end();
            if (swap.lived.size()) iter = std::prev(iter);
            while (iter != swap.lived.end()) {
                auto var = iter->second;
                iter = iter==swap.lived.begin() ? swap.lived.end() : std::prev(iter);
                if (var->tflag == swap_timestamp)
                    continue;
                ASSERT(var->mem_ptr) << x << var;
                if (!is_cpu) {
                    // try move to cpu
                    if (!move_with_swap(var, cpu_allocator, false))
                        swap_to_disk(var, swap);
                } else
                    swap_to_disk(var, swap);
                allocator->gc();
                if (allocator->used_memory + allocator->unused_memory + x->size <= limit || allocator->unused_memory >= unused_target) break;
            }
            if (!(allocator->used_memory + allocator->unused_memory + x->size <= limit || allocator->unused_memory >= unused_target)) {
                display_memory_info();
                LOGw << "unable to alloc var" << x;
            }
        }
    }
    if (x->alloc(allocator)) {
        swap.lived[{x->size, x->id}] = x;
        return true;
    }
    return false;
}

void free_with_swap(Var* x) {
    if (x->flag(VarFlags::_is_swapped)) {
        string path = swap_file_path(x);
        if (remove(path.c_str()) != 0)
            LOGe << "failed to remove swap file" << path << x->shape << x->dtype();
    } else {
        if (!x->mem_ptr) return;
        auto& swap = swaps[x->allocator];
        auto iter = swap.lived.find({x->size, x->id});
        if (iter != swap.lived.end())
            swap.lived.erase(iter);
        x->allocator->free(x->mem_ptr, x->size, x->allocation);
        x->mem_ptr = nullptr;
        x->allocator = nullptr;
        x->allocation = 0;
    }
}

bool move_with_swap(Var* x, Allocator* allocator, bool force) {
    if (allocator == x->allocator) return true;
    // same reason as swap_to_disk: moving one member of a share group breaks
    // the alias for the rest (see migrate_group in mem/allocator.cc)
    ASSERT(!x->share_next) << "cannot move an aliased var with swap" << x;
    swap_total += x->size;
    Allocation allocation(x->mem_ptr, x->allocation, x->size, x->allocator);
    x->mem_ptr = nullptr;
    x->allocator = nullptr;
    x->allocation = 0;
    if (!alloc_with_swap(x, allocator, force)) {
        x->mem_ptr = allocation.ptr;
        x->allocator = allocation.allocator;
        x->allocation = allocation.allocation;
        allocation.ptr = nullptr;
        allocation.allocation = 0;
        return false;
    }
    if (x->flag(VarFlags::_is_swapped)) {
        string path = swap_file_path(x);
        #ifdef HAS_CUDA
        if (x->allocator->is_cuda()) {
            int64 buf_size = std::min(x->size, SWAP_BUF_SIZE);
            std::unique_ptr<char[]> buf(new char[buf_size]);
            char* buffer = buf.get();
            auto* memptr = (char*)x->mem_ptr;
            auto* fd = fopen(path.c_str(), "rb");
            CHECK(fd) << "swap file open failed:" << path << x;
            for (int64 i=0; i<x->size; i+=SWAP_BUF_SIZE) {
                int64 cp_size = std::min(x->size-i, SWAP_BUF_SIZE);
                auto res = fread(buffer, cp_size, 1, fd);
                if (res != 1) {
                    fclose(fd);
                    LOGf << "swap file read failed" << path << x;
                }
                checkCudaErrors(cudaMemcpy(memptr+i, buffer, cp_size, cudaMemcpyHostToDevice));
            }
            fclose(fd); 
        } else
        #endif
        {
            auto* fd = fopen(path.c_str(), "rb");
            CHECK(fd) << "swap file open failed:" << path << x;
            auto res = fread(x->mem_ptr, x->size, 1, fd);
            CHECK(res==1);
            fclose(fd); 
        }
            
        if (remove(path.c_str()) != 0)
            LOGe << "failed to remove swap file" << path << x->shape << x->dtype();
        x->set_flag(VarFlags::_is_swapped, 0);
    } else {
        #ifdef HAS_CUDA
        if (x->allocator->is_cuda()) {
            if (allocation.allocator->is_cuda())
                checkCudaErrors(cudaMemcpy(x->mem_ptr, allocation.ptr, x->size, cudaMemcpyDeviceToDevice));
            else
                checkCudaErrors(cudaMemcpy(x->mem_ptr, allocation.ptr, x->size, cudaMemcpyHostToDevice));
        } else
        if (allocation.allocator->is_cuda()) {
            checkCudaErrors(cudaMemcpy(x->mem_ptr, allocation.ptr, x->size, cudaMemcpyDeviceToHost));
        } else
        #endif
        {
            std::memcpy(x->mem_ptr, allocation.ptr, x->size);
        }
    }
    if (allocation.ptr) {
        auto& swap = swaps[allocation.allocator];
        auto iter = swap.lived.find({x->size, x->id});
        if (iter != swap.lived.end())
            swap.lived.erase(iter);
    }
    return true;
}

void registe_swap(Var* x) {
    auto& swap = swaps[x->allocator];
    swap.lived[{x->size, x->id}] = x;
}

}
