// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "profiler/memory_checker.h"

int virt_to_phys_user(uintptr_t* paddr, uintptr_t vaddr);

namespace jittor {
MemoryChecker::MemoryChecker(Cache* tlb, vector<Cache*> caches, size_t page_size, size_t vtop)
: tlb(tlb), caches(caches), page_size(page_size), vtop(vtop) {
    clear();
}

MemoryChecker::~MemoryChecker() {
    delete tlb;
    for (int i = 0; i < (int)caches.size(); ++i)
        delete caches[i];
}

string MemoryChecker::get_replace_strategy(int id) {
    if (id == 0)
        return "DefaultReplacementCache";
    if (id == 1)
        return "LRUCache";
    return "DefaultReplacementCache";
}

void MemoryChecker::clear() {
    check_times = 0;
    tlb->clear();
    for (int i = 0; i < (int)caches.size(); ++i)
        caches[i]->clear();
}

void MemoryChecker::print_miss() {
    LOGi << "Total:" << check_times;
    LOGi << "TLB Misses:" << tlb->miss_time;
    for (int i = 0; i < (int)caches.size(); ++i)
        LOGi << "L" >> (i+1) << "Cache Misses:" << caches[i]->miss_time;
}

void MemoryChecker::check_hit(size_t vaddr) {
    size_t paddr;
    if (vtop)
        CHECK(virt_to_phys_user(&paddr, vaddr)==0)
            << "FAILED to translate vaddr to paddr";
    else
        paddr = vaddr;
    ++check_times;
    for (int i = 0; i < (int)caches.size(); ++i)
        if (caches[i]->check_hit(paddr))
            break;
    tlb->check_hit(size_t(vaddr)/page_size);
}

} //jittor