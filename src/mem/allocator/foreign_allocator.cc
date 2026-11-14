// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <unordered_map>
#include <memory>
#include <mutex>
#include "mem/allocator/foreign_allocator.h"
#include "core/var.h"
#include "runtime/async_exec.h"

namespace jittor {

struct ForeignAllocation {
    std::function<void()> del_func;
    int64 cnt;
    ForeignAllocation(std::function<void()>&& del_func)
        : del_func(std::move(del_func)), cnt(1) {}
};

ForeignAllocator foreign_allocator;

ForeignAllocator& get_foreign_allocator(int device_id) {
    if (device_id < 0) return foreign_allocator;
    static std::mutex pool_mutex;
    static std::unordered_map<int, std::unique_ptr<ForeignAllocator>> pool;
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto iter = pool.find(device_id);
    if (iter != pool.end()) return *iter->second;
    auto* p = new ForeignAllocator(device_id);
    pool[device_id] = std::unique_ptr<ForeignAllocator>(p);
    return *p;
}

const char* ForeignAllocator::name() const {return "foreign";}

void* ForeignAllocator::alloc(size_t size, size_t& allocation) {
    return nullptr;
}

void ForeignAllocator::free(void* mem_ptr, size_t size, const size_t& allocation) {
    auto a = (ForeignAllocation*)allocation;
    a->cnt--;
    if (!a->cnt) {
        auto release = std::move(a->del_func);
        delete a;
        // DLPack and NumPy owners can acquire the GIL in their deleters.
        // The async worker releases liveness under the graph lock, while a
        // Python deallocator waits for that lock with the GIL held. Keep the
        // producer alive until the caller can release it, as for Python tapes.
        if (!async_exec_defer_python_release(std::move(release)))
            release();
    }
}

void make_foreign_allocation(Allocation& a, void* ptr, size_t size, std::function<void()>&& del_func,
        Allocator* target) {
    auto fa = new ForeignAllocation(std::move(del_func));
    a.allocator = target ? target : &foreign_allocator;
    a.allocation = (size_t)fa;
    a.ptr = ptr;
    a.size = size;
}

bool ForeignAllocator::share_with(size_t size, size_t allocation, size_t offset) {
    ((ForeignAllocation*)allocation)->cnt++;
    return true;
}

} // jittor
