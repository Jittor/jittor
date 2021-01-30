// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <pthread.h>
#include <sys/mman.h>
#include <cstring>
#include "common.h"

namespace jittor {

struct RingBuffer {

    struct Mutex {
        pthread_mutex_t m;
        inline Mutex(bool multiprocess=0) {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            if (multiprocess)
                pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            ASSERT(0 == pthread_mutex_init((pthread_mutex_t*)&m, &attr));
        }
        
        inline ~Mutex() {
            pthread_mutex_destroy(&m);
        }

        inline void lock() {
            pthread_mutex_lock(&m);
        }

        inline void unlock() {
            pthread_mutex_unlock(&m);
        }
    };

    struct Cond {
        pthread_cond_t cv;
        inline Cond(bool multiprocess=0) {
            pthread_condattr_t attr;
            pthread_condattr_init(&attr);
            if (multiprocess)
                pthread_condattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            ASSERT(0 == pthread_cond_init((pthread_cond_t*)&cv, &attr));
        }
        
        inline ~Cond() {
            // a dirty hack
            // ref: https://stackoverflow.com/questions/20439404/pthread-conditions-and-process-termination
            // cv.__data.__wrefs = 0;
            cv.__data = {0};
            pthread_cond_destroy(&cv);
        }

        inline void wait(Mutex& m) {
            pthread_cond_wait(&cv, &m.m);
        }

        inline void notify() {
            pthread_cond_signal(&cv);
        }
    };

    struct MutexScope {
        Mutex* m;
        inline MutexScope(Mutex& m) : m(&m) { m.lock(); }
        inline ~MutexScope() { m->unlock(); }
    };

    uint64 size;
    uint64 size_mask;
    uint64 size_bit;
    volatile uint64 l;
    volatile uint64 r;
    volatile bool is_wait;
    volatile bool is_stop;
    bool is_multiprocess;
    Mutex m;
    Cond cv;
    char _ptr;

    RingBuffer(uint64 size, bool multiprocess=false);
    ~RingBuffer();
    void stop();
    static RingBuffer* make_ring_buffer(uint64 size, bool multiprocess);
    static void free_ring_buffer(RingBuffer* rb);

    inline void clear() { l = r = is_stop = 0; }

    inline void wait() {
        if (is_stop) {
            throw std::runtime_error("stop");
        }
        {
            MutexScope _(m);
            if (is_wait) {
                cv.notify();
                is_wait = 0;
            }
            is_wait = 1;
            cv.wait(m);
        }
    }

    inline void notify() {
        MutexScope _(m);
        cv.notify();
        is_wait = 0;
    }

    inline void push(uint64 size, uint64& __restrict__ offset) {
        auto rr = offset;
        auto rr_next = rr + size;
        auto c1 = rr >> size_bit;
        auto c2 = (rr_next-1) >> size_bit;
        if (c1 != c2) {
            // if cross boundary
            rr = c2 << size_bit;
            rr_next = rr + size;
        }
        CHECK(rr_next <= r+this->size) << "Buffer size too small, please increase buffer size. Current size:"
            << this->size << "Required size:" << rr_next - r;
        while (rr_next > l + this->size) {
            wait();
        }
        offset = rr_next;
    }

    inline void commit_push(uint64 offset) {
        r = offset;
        if (is_wait)
            notify();
    }

    inline void pop(uint64 size, uint64& __restrict__ offset) {
        auto ll = offset;
        auto ll_next = ll + size;
        auto c1 = ll >> size_bit;
        auto c2 = (ll_next-1) >> size_bit;
        if (c1 != c2) {
            // if cross boundary
            ll = c2 << size_bit;
            ll_next = ll + size;
        }
        while (ll_next > r) {
            ASSERT(size<=this->size);
            wait();
        }
        offset = ll_next;
    }

    inline void commit_pop(uint64 offset) {
        l = offset;
        if (is_wait)
            notify();
    }

    inline uint64 push(uint64 size) { 
        auto offset = r;
        push(size, offset);
        return offset;
    }
    inline uint64 pop(uint64 size) {
        auto offset = l;
        pop(size, offset); 
        return offset;
    }

    inline char* get_ptr(uint64 size, uint64 offset) { return ((&_ptr)+((offset-size)&size_mask)); }

    template<class T>
    inline T& get(uint64 offset) { return *(T*)((&_ptr)+((offset-sizeof(T))&size_mask)); }

    template<class T>
    inline void push_t(const T& data, uint64& __restrict__ offset) {
        push(sizeof(T), offset);
        get<T>(offset) = data;
    }

    template<class T>
    inline T& pop_t(uint64& __restrict__ offset) {
        pop(sizeof(T), offset);
        return get<T>(offset);
    }

    inline void push_string(const string& data, uint64& __restrict__ offset) {
        push_t<int64>(data.size(), offset);
        push(data.size(), offset);
        auto ptr = get_ptr(data.size(), offset);
        std::memcpy(ptr, data.c_str(), data.size());
    }

    inline string pop_string(uint64& __restrict__ offset) {
        auto size = pop_t<int64>(offset);
        pop(size, offset);
        auto ptr = get_ptr(size, offset);
        return string(ptr, size);
    }

    template<class T>
    inline void push_t(const T& data) {
        auto offset = push(sizeof(T));
        get<T>(offset) = data;
        commit_push(offset);
    }

    template<class T>
    inline T pop_t() {
        auto offset = pop(sizeof(T));
        T data = get<T>(offset);
        commit_pop(offset);
        return data;
    }
};

}
