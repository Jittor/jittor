// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <typeinfo>
#include "runtime/device.h"
#include "runtime/backend.h"

#ifdef HAS_ACCELERATOR
#include "mem/allocator/cuda_dual_allocator.h"
#endif
#include "mem/allocator/stat_allocator.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/nfef_allocator.h"
#include "mem/allocator/temp_allocator.h"
#include "mem/swap.h"
#include "runtime/traversal_epoch.h"
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

Allocator* cpu_allocator = setup_allocator<SFRLAllocator>(
    backend_raw_allocator({BackendId::Cpu, 0}, BackendMemoryKind::Device));

DECLARE_FLAG(int, use_cuda_managed_allocator);

DEFINE_FLAG_WITH_SETTER(int, use_cuda_host_allocator, 1, "use cuda host allocator for cpu memory globally");

void setter_use_cuda_host_allocator(const int& old_value, const int& value) {
    #ifdef HAS_ACCELERATOR
    // `use_cuda_host_allocator = value;` used to be here so that the
    // get_allocator() below could see the new value. The macro assigns first
    // now, so it already does.
    auto use_cuda_bk = runtime_use_cuda();
    runtime_device_state().use_cuda = 0;
    cpu_allocator = get_allocator();
    runtime_device_state().use_cuda = use_cuda_bk;
    #endif
}

extern int64 sfrl_large_block_size_device;

bool use_pinned_host_memory() {
    if (use_cuda_host_allocator) return true;
#ifdef HAS_ACCELERATOR
    if (runtime_use_cuda())
        return backend_ops(accelerator_backend_id()).execution.requires_pinned_host_storage;
#endif
    return false;
}

Allocator* get_array_host_allocator() {
#ifdef HAS_ACCELERATOR
    if (runtime_use_cuda()
            && backend_ops(accelerator_backend_id()).execution.requires_pinned_host_storage)
        return get_allocator(-1, false);
#endif
    return cpu_allocator;
}

Allocator* get_allocator(bool temp_allocator) {
    int device = -1;
#ifdef HAS_ACCELERATOR
    if (runtime_use_cuda()) device = current_device();
#endif
    return get_allocator(device, temp_allocator);
}

Allocator* get_allocator(int device, bool temp_allocator) {
    Allocator* allocator = nullptr;
    if (runtime_use_cuda() && sfrl_large_block_size_device >= (1ll<<40)) {
        // if super large block is used, don't use
        // temp allocator
        temp_allocator = false;
    }
#ifdef HAS_ACCELERATOR
    if (runtime_use_cuda() && device >= 0 && !allocator) {
        LOGvv << "Using cuda allocator of device" << device;
        allocator = backend_raw_allocator({accelerator_backend_id(), device},
            use_cuda_managed_allocator ? BackendMemoryKind::Managed : BackendMemoryKind::Device);
    } else
#endif
    {
        allocator = backend_raw_allocator({BackendId::Cpu, 0},
            use_pinned_host_memory() ? BackendMemoryKind::Pinned : BackendMemoryKind::Device);
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
    // a zero-sized var carries no data, so leaving its share group *is* the
    // whole migration; the free below drops its reference to the parent block
    if (PREDICT_BRANCH_NOT_TAKEN(var->share_next != nullptr))
        share_group_unlink(var);
    Allocation target(allocator, 0);
    if (var->mem_ptr && var->allocator)
        var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = target.ptr;
    var->allocation = target.allocation;
    var->allocator = target.allocator;
    target.ptr = nullptr;
}

#ifdef HAS_ACCELERATOR
// Move every var that shares one allocation, in one step.
//
// Var::alloc's share_with branch leaves a child indistinguishable from its
// parent, so the single-var path below -- fresh block, memcpy, free the old
// one -- gives the migrated var a private copy and drops the alias without a
// word: an in-place write through either var stops being visible through the
// other. The ring built in var.cc is what makes the group visible here; the
// group moves into one new block that keeps every member's relative offset.
static bool migrate_group(Var* var, Allocator* allocator, bool to_gpu) {
    auto* old_allocator = var->allocator;
    auto old_allocation = var->allocation;
    vector<Var*> members;
    {
        // Collect the ring before pruning it: unlinking rewires what we walk.
        vector<Var*> ring;
        Var* p = var;
        do { ring.push_back(p); p = p->share_next; } while (p != var);
        for (auto* v : ring) {
            if (v->mem_ptr && v->allocator == old_allocator
                    && v->allocation == old_allocation) {
                members.push_back(v);
                continue;
            }
            // A var's memory can be replaced without going through
            // free_var_mem -- ArrayOp::run does exactly that -- and such a var
            // left the group when that happened; only the ring still says
            // otherwise. It is not an alias any more, so drop it.
            LOGvvvv << "share ring: dropping stale member" << v;
            share_group_unlink(v);
        }
    }
    if (members.size() < 2) {
        // nothing is actually aliased: let the caller take the plain path
        share_group_unlink(var);
        return false;
    }
    if (!allocator->can_share()) {
        // The target cannot express one block held by several vars, so this
        // group cannot survive the move at all. That is what has always
        // happened here; the difference is that it no longer happens quietly.
        // Only the debug allocator configurations reach this
        // (use_sfrl_allocator=0, use_nfef_allocator=1): the default stack and
        // cpu_allocator are both SFRL, which can share.
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOGw << "allocator" << allocator->name() << "cannot hold a share"
                << "group, so migrating an aliased var breaks the alias:"
                << "this allocator configuration cannot represent"
                << "Var::share_with. Writes through one alias will not be"
                << "visible through the other.";
        }
        share_group_unlink(var);
        return false;
    }
    char* base = (char*)var->mem_ptr;
    char* end = base + var->size;
    for (auto* v : members) {
        char* vp = (char*)v->mem_ptr;
        if (vp < base) base = vp;
        if (vp + v->size > end) end = vp + v->size;
    }
    size_t total = end - base;
    vector<size_t> offsets(members.size());
    for (size_t i=0; i<members.size(); i++)
        offsets[i] = (char*)members[i]->mem_ptr - base;
    Allocation a(allocator, total);
    backend_copy(a.ptr, allocation_device(allocator), base,
                 allocation_device(old_allocator), total);
    // Take one reference per extra member before touching any var, so a target
    // that cannot express sharing fails with the group still intact.
    for (size_t i=1; i<members.size(); i++)
        CHECK(allocator->share_with(members[i]->size, a.allocation))
            << "allocator" << allocator->name()
            << "cannot hold a share group; migrating one var of an aliased "
               "group would silently unshare it";
    for (size_t i=0; i<members.size(); i++) {
        auto* v = members[i];
        v->mem_ptr = (char*)a.ptr + offsets[i];
        v->allocation = a.allocation;
        v->allocator = allocator;
    }
    for (size_t i=0; i<members.size(); i++)
        old_allocator->free(base + offsets[i], members[i]->size, old_allocation);
    a.ptr = nullptr;
    return true;
}
#endif

void migrate_to_cpu(Var* var, Allocator* allocator) {
    #ifdef HAS_ACCELERATOR
    if (!use_cuda_managed_allocator)
        allocator = cpu_allocator;
    #endif
    if (var->size == 0) {
        migrate_empty_var(var, allocator);
        return;
    }
    if (save_mem) {
        if (swap_timestamp != var->tflag) {
            TraversalEpoch epoch("migrate_to_cpu");
            swap_timestamp = epoch.stamp;
            epoch.mark(var);
            move_with_swap(var, cpu_allocator, true);
            return;
        }
        move_with_swap(var, cpu_allocator, true);
        return;
    }
    #ifdef HAS_ACCELERATOR
    // An aliased var can only move together with the rest of its group; if it
    // turns out not to be aliased after all, migrate_group says so and the
    // plain path below runs instead (see migrate_group).
    if (PREDICT_BRANCH_NOT_TAKEN(var->share_next != nullptr)
        && var->mem_ptr && var->allocator->is_cuda()
        && (var->allocator == &delay_free || !use_cuda_managed_allocator)
        && migrate_group(var, allocator, false))
        return;
    if (var->allocator == &delay_free) {
        var->allocator = allocator;
        delay_free.migrate_to_cpu(
            var->mem_ptr, var->allocation, var->size, var->allocator
        );
    } else
    if (!use_cuda_managed_allocator) {
        if (!var->allocator->is_cuda()) return;
        // must be a device allocator. Issue the copy with the var's own
        // device current, so it is ordered after the kernels that produced it
        // rather than after whatever the current device happens to be running.
        Allocation a(allocator, var->size);
        backend_copy(a.ptr, allocation_device(allocator), var->mem_ptr,
                     allocation_device(var->allocator), var->size);
        var->allocator->free(var->mem_ptr, var->size, var->allocation);
        var->mem_ptr = a.ptr;
        var->allocation = a.allocation;
        var->allocator = a.allocator;
        a.ptr = nullptr;
    }
    #endif
}


void migrate_to_gpu(Var* var, Allocator* allocator) {
    #ifdef HAS_ACCELERATOR
    // only happend when not using use_cuda_managed_allocator
    if (var->size == 0) {
        migrate_empty_var(var, allocator);
        return;
    }
    if (save_mem) {
        if (swap_timestamp != var->tflag) {
            TraversalEpoch epoch("migrate_to_gpu");
            swap_timestamp = epoch.stamp;
            epoch.mark(var);
            move_with_swap(var, allocator, true);
            return;
        }
        move_with_swap(var, allocator, true);
        return;
    }
    // aliased: all or nothing (see migrate_group)
    if (PREDICT_BRANCH_NOT_TAKEN(var->share_next != nullptr)
        && var->mem_ptr && migrate_group(var, allocator, true))
        return;
    Allocation a(allocator, var->size);
    // Upload onto the pool's own device: a cache hit inside the pool skips
    // cudaMalloc, so the current device is not guaranteed to be right here.
    backend_copy(a.ptr, allocation_device(allocator), var->mem_ptr,
                 allocation_device(var->allocator), var->size);
    var->allocator->free(var->mem_ptr, var->size, var->allocation);
    var->mem_ptr = a.ptr;
    var->allocation = a.allocation;
    var->allocator = a.allocator;
    a.ptr = nullptr;
    #endif
}

} // jittor
