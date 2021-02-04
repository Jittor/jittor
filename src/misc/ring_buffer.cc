// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include "common.h"
#include "misc/ring_buffer.h"

namespace jittor {

RingBuffer::RingBuffer(uint64 size, bool multiprocess) : m(multiprocess), cv(multiprocess) {
    int i=0;
    for (;(1ll<<i)<size;i++);
    size_mask = (1ll<<i)-1;
    this->size = size_mask+1;
    size_bit = i;
    l = r = is_wait = is_stop = 0;
    is_multiprocess = multiprocess;
}

void RingBuffer::stop() {
    MutexScope _(m);
    is_stop = 1;
    cv.notify();
}

RingBuffer::~RingBuffer() {
    stop();
}


RingBuffer* RingBuffer::make_ring_buffer(uint64 size, bool multiprocess) {
    int i=0;
    for (;(1ll<<i)<size;i++);
    uint64 size_mask = (1ll<<i)-1;
    size = size_mask+1;
    uint64 total_size = sizeof(RingBuffer) + size;
    void* ptr = multiprocess ?
        // mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0) :
        mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0) :
        // mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, -1, 0) :
        (void*)malloc(total_size);
    std::memset(ptr, 0, total_size);
    auto rb = (RingBuffer*)ptr;
    new (rb) RingBuffer(size, multiprocess);
    return rb;
}

void RingBuffer::free_ring_buffer(RingBuffer* rb) {
    uint64 total_size = sizeof(RingBuffer) + rb->size;
    auto is_multiprocess = rb->is_multiprocess;
    rb->~RingBuffer();
    if (is_multiprocess) {
        munmap(rb, total_size);
    } else {
        free((void*)rb);
    }
}

// test

JIT_TEST(ring_buffer_benchmark) {
    size_t n = 1ll << 20;
    size_t size = 1<<15;
    // size_t n = 1ll << 30;
    // size_t size = 1<<20;
    // size_t n = 1ll << 10;
    // size_t size = 1<<5;
    RingBuffer* rb = RingBuffer::make_ring_buffer(size, 0);
    std::thread p([&]() {
        for (size_t i=0; i<n; i++) {
            rb->push_t<int>(i);
        }
    });
    auto start = std::chrono::high_resolution_clock::now();
    size_t s = 0;
    for (size_t i=0; i<n; i++) {
        auto x = rb->pop_t<int>();
        s += x;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    auto tt =  std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    p.join();
    expect_error([&]() { rb->push(size+1); });
    RingBuffer::free_ring_buffer(rb);

    LOGi << tt << tt*1.0/n;
    LOGi << s << (n*(n-1)/2); 
    ASSERTop(s,==,(n*(n-1)/2));
    ASSERTop(tt*1.0/n,<=,100);
}

}
