// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <functional>
#include <condition_variable>

// fast multi writer single reader list
#define MWSR_LIST(name, T) \
namespace mwsr_list_ ## name { \
    using std::list; \
    using std::vector; \
    using std::function; \
     \
    typedef T mylist_t; \
    struct ThreadList { \
        list<mylist_t> values; \
        std::mutex mutex; \
    }; \
    list<std::unique_ptr<ThreadList>> glist; \
    std::mutex glist_mutex; \
    std::condition_variable cv; \
    std::mutex mm; \
    std::atomic<size_t> pending(0); \
    bool _stop; \
    bool _flush; \
     \
    void clear() { \
        { \
            std::lock_guard<std::mutex> lk(glist_mutex); \
            for (auto& state : glist) { \
                std::lock_guard<std::mutex> state_lk(state->mutex); \
                state->values.clear(); \
            } \
            pending.store(0, std::memory_order_release); \
        } \
        { \
            std::lock_guard<std::mutex> lk(mm); \
            _stop = false; \
            _flush = false; \
        } \
    } \
     \
    void flush() { \
        { \
            std::lock_guard<std::mutex> lk(mm); \
            _flush = true; \
        } \
        cv.notify_one(); \
    } \
     \
    void stop() { \
        { \
            std::lock_guard<std::mutex> lk(mm); \
            _stop = true; \
        } \
        cv.notify_one(); \
    } \
     \
    void init() { \
        clear(); \
    } \
     \
    ThreadList* create_tlist() { \
        std::lock_guard<std::mutex> lk(glist_mutex); \
        glist.emplace_back(new ThreadList); \
        return glist.back().get(); \
    } \
     \
    thread_local ThreadList* tlist = create_tlist(); \
     \
    void push(mylist_t &&s) { \
        { \
            std::lock_guard<std::mutex> lk(tlist->mutex); \
            tlist->values.emplace_back(std::move(s)); \
            pending.fetch_add(1, std::memory_order_release); \
        } \
        cv.notify_one(); \
    } \
     \
    void reduce(function<void(const mylist_t&)> func, function<void()> flush_func) { \
        vector<ThreadList*> states; \
        while (1) { \
            { \
                std::lock_guard<std::mutex> lk(glist_mutex); \
                if (states.size() != glist.size()) { \
                    states.clear(); \
                    for (auto& state : glist) \
                        states.push_back(state.get()); \
                } \
            } \
            bool found = false; \
            for (auto state : states) { \
                list<mylist_t> batch; \
                { \
                    std::lock_guard<std::mutex> lk(state->mutex); \
                    batch.splice(batch.end(), state->values); \
                } \
                if (!batch.empty()) { \
                    pending.fetch_sub(batch.size(), std::memory_order_acq_rel); \
                    for (auto& value : batch) \
                        func(value); \
                    found = true; \
                } \
            } \
            if (!found) { \
                bool should_flush = false; \
                bool should_stop = false; \
                std::unique_lock<std::mutex> lk(mm); \
                if (pending.load(std::memory_order_acquire) == 0 && _flush) { \
                    _flush = false; \
                    should_flush = true; \
                } \
                if (pending.load(std::memory_order_acquire) == 0 && _stop) { \
                    should_stop = true; \
                } \
                if (!should_flush && !should_stop) { \
                    cv.wait(lk, [] { \
                        return _stop || _flush || \
                            pending.load(std::memory_order_acquire) != 0; \
                    }); \
                } \
                lk.unlock(); \
                if (should_flush) \
                    flush_func(); \
                if (should_stop) \
                    break; \
            } \
        } \
        init(); \
    } \
} // mwsr_list
