// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif
#include <atomic>
#include <cstring>
#include <ctime>
#include <cerrno>
#include "common.h"

namespace jittor {

struct RingBuffer {

#ifdef _MSC_VER
    struct Mutex {
        HANDLE handle;
        inline Mutex(bool multiprocess=0) {
        }
        
        inline void lock() {
        }

        inline void unlock() {
        }
        inline ~Mutex() {
        }
    };
    struct MutexScope {
        Mutex* m;
        inline MutexScope(Mutex& m) : m(&m) { m.lock(); }
        inline ~MutexScope() { m->unlock(); }
    };

    struct Cond {
        inline Cond(bool multiprocess=0) {
        }

        inline void wait(MutexScope& m) {
        }

        inline void notify() {
        }
    };
#else
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
    struct MutexScope {
        Mutex* m;
        inline MutexScope(Mutex& m) : m(&m) { m.lock(); }
        inline ~MutexScope() { m->unlock(); }
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
            #ifdef __linux__
            cv.__data = {0};
            #endif
            pthread_cond_destroy(&cv);
        }

        inline void wait(MutexScope& m) {
            pthread_cond_wait(&cv, &m.m->m);
        }

        inline void notify() {
            pthread_cond_signal(&cv);
        }
    };
#endif

    uint64 size;
    uint64 size_mask;
    uint64 size_bit;
    alignas(64) std::atomic<uint64> l;
    std::atomic<bool> is_push_wait;
    uint64 r_cache;
    alignas(64) std::atomic<uint64> r;
    std::atomic<bool> is_pop_wait;
    uint64 l_cache;
    std::atomic<bool> is_stop;
    bool is_multiprocess;
    Mutex m;
    Cond push_cv;
    Cond pop_cv;
    char _ptr;

    RingBuffer(uint64 size, bool multiprocess=false);
    ~RingBuffer();
    void stop();
    static RingBuffer* make_ring_buffer(uint64 size, bool multiprocess, uint64 buffer=0, bool init=true);
    static void free_ring_buffer(RingBuffer* rb, uint64 buffer=0, bool init=true);

    inline void clear() {
        l.store(0, std::memory_order_release);
        r.store(0, std::memory_order_release);
        l_cache = 0;
        r_cache = 0;
        is_push_wait.store(false, std::memory_order_release);
        is_pop_wait.store(false, std::memory_order_release);
        is_stop.store(false, std::memory_order_release);
    }

    inline void wait_push(uint64 offset) {
        MutexScope _(m);
        while (true) {
            is_push_wait.store(true, std::memory_order_seq_cst);
            auto current_l = l.load(std::memory_order_seq_cst);
            l_cache = current_l;
            if (offset <= current_l + size)
                break;
            if (is_stop.load(std::memory_order_acquire)) {
                is_push_wait.store(false, std::memory_order_seq_cst);
                throw std::runtime_error("stop");
            }
            push_cv.wait(_);
        }
        is_push_wait.store(false, std::memory_order_seq_cst);
    }

    inline void wait_pop(uint64 offset) {
        MutexScope _(m);
        while (true) {
            is_pop_wait.store(true, std::memory_order_seq_cst);
            auto current_r = r.load(std::memory_order_seq_cst);
            r_cache = current_r;
            if (offset <= current_r)
                break;
            if (is_stop.load(std::memory_order_acquire)) {
                is_pop_wait.store(false, std::memory_order_seq_cst);
                throw std::runtime_error("stop");
            }
            pop_cv.wait(_);
        }
        is_pop_wait.store(false, std::memory_order_seq_cst);
    }

    inline void wait_pop_for(uint64 offset, uint64 timeout_ms) {
        MutexScope _(m);
        timespec deadline;
#ifndef _MSC_VER
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += timeout_ms / 1000;
        deadline.tv_nsec += (timeout_ms % 1000) * 1000000;
        if (deadline.tv_nsec >= 1000000000) {
            ++deadline.tv_sec;
            deadline.tv_nsec -= 1000000000;
        }
#endif
        while (true) {
            is_pop_wait.store(true, std::memory_order_seq_cst);
            auto current_r = r.load(std::memory_order_seq_cst);
            r_cache = current_r;
            if (offset <= current_r)
                break;
            if (is_stop.load(std::memory_order_acquire)) {
                is_pop_wait.store(false, std::memory_order_seq_cst);
                throw std::runtime_error("stop");
            }
#ifdef _MSC_VER
            // Windows keeps the legacy condition-variable path; the bounded
            // worker-death diagnostic is currently Linux-only.
            pop_cv.wait(_);
#else
            auto status = pthread_cond_timedwait(
                &pop_cv.cv, &m.m, &deadline);
            if (status == ETIMEDOUT) {
                is_pop_wait.store(false, std::memory_order_seq_cst);
                throw std::runtime_error("ring buffer pop timed out");
            }
            if (status != 0 && status != EINTR)
                throw std::runtime_error("ring buffer pop wait failed");
#endif
        }
        is_pop_wait.store(false, std::memory_order_seq_cst);
    }

    inline void notify_push() {
        MutexScope _(m);
        is_push_wait.store(false, std::memory_order_seq_cst);
        push_cv.notify();
    }

    inline void notify_pop() {
        MutexScope _(m);
        is_pop_wait.store(false, std::memory_order_seq_cst);
        pop_cv.notify();
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
        auto current_r = r.load(std::memory_order_relaxed);
        CHECK(rr_next <= current_r+this->size) << "Buffer size too small, please increase buffer size. Current size:"
            << this->size << "Required size:" << rr_next - current_r;
        if (rr_next > l_cache + this->size) {
            l_cache = l.load(std::memory_order_acquire);
            if (rr_next > l_cache + this->size)
                wait_push(rr_next);
        }
        offset = rr_next;
    }

    inline void commit_push(uint64 offset) {
        r.store(offset, std::memory_order_seq_cst);
        if (is_pop_wait.load(std::memory_order_seq_cst))
            notify_pop();
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
        ASSERT(size<=this->size);
        if (ll_next > r_cache) {
            r_cache = r.load(std::memory_order_acquire);
            if (ll_next > r_cache)
                wait_pop(ll_next);
        }
        offset = ll_next;
    }

    inline void commit_pop(uint64 offset) {
        l.store(offset, std::memory_order_seq_cst);
        if (is_push_wait.load(std::memory_order_seq_cst))
            notify_push();
    }

    inline uint64 push(uint64 size) { 
        auto offset = r.load(std::memory_order_relaxed);
        push(size, offset);
        return offset;
    }
    inline uint64 pop(uint64 size) {
        auto offset = l.load(std::memory_order_relaxed);
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
