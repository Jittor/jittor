// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <list>
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
    list<list<mylist_t>> glist; \
    list<list<mylist_t>::iterator> glist_iter; \
    std::mutex glist_mutex; \
    std::condition_variable cv; \
    std::mutex mm; \
    bool _stop; \
    bool _flush; \
     \
    void clear() { \
        std::lock_guard<std::mutex> lk(glist_mutex); \
        glist.clear(); \
        glist_iter.clear(); \
        _stop = false; \
        _flush = false; \
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
        std::lock_guard<std::mutex> lk(glist_mutex); \
        _stop = false; \
        _flush = false; \
        auto titer = glist_iter.begin(); \
        for (auto& tlist : glist) { \
            tlist.clear(); \
            *titer = tlist.end(); \
            titer ++; \
        } \
    } \
     \
    list<mylist_t>* create_tlist() { \
        std::lock_guard<std::mutex> lk(glist_mutex); \
        glist.emplace_back(); \
        auto tlist = &glist.back(); \
        glist_iter.push_back(tlist->end()); \
        return tlist; \
    } \
     \
    thread_local list<mylist_t>* tlist = create_tlist(); \
     \
    void push(mylist_t &&s) { \
        tlist->emplace_back(move(s)); \
        cv.notify_one(); \
    } \
     \
    void reduce(function<void(const mylist_t&)> func, function<void()> flush_func) { \
        thread_local vector<list<mylist_t>*> gvlist; \
        thread_local vector<list<mylist_t>::iterator*> gvlist_iter; \
        gvlist.clear(); \
        gvlist_iter.clear(); \
        int stop2=0; \
        int flush2=0; \
        while (1) { \
            int found = 0; \
            if (gvlist.size() != glist.size()) { \
                std::lock_guard<std::mutex> lk(glist_mutex); \
                gvlist.clear(); \
                gvlist_iter.clear(); \
                for (auto &tlist : glist) \
                    gvlist.push_back(&tlist); \
                for (auto &tlist_iter : glist_iter) \
                    gvlist_iter.push_back(&tlist_iter); \
            } \
             \
            auto list_iter = gvlist_iter.begin(); \
            for (auto tlist : gvlist) { \
                auto& last = **list_iter; \
                if (last == tlist->end()) { \
                    last = tlist->begin(); \
                    if (last != tlist->end()) { \
                        func(*last); \
                        found++; \
                    } \
                } \
                while (last != tlist->end()) { \
                    auto nlast = next(last); \
                    if (nlast != tlist->end()) { \
                        func(*nlast); \
                        last = nlast; \
                        tlist->pop_front(); \
                        found++; \
                    } else break; \
                } \
                list_iter ++; \
            } \
            if (!found) { \
                std::unique_lock<std::mutex> lk(mm); \
                if (_flush) { \
                    _flush = false; \
                    flush2 = 1; \
                    lk.unlock(); \
                    continue; \
                } \
                if (flush2) { \
                    flush2 = 0; \
                    flush_func(); \
                } \
                if (_stop) { \
                    if (stop2>0) { \
                        lk.unlock(); \
                        break; \
                    } else { \
                        stop2 ++; \
                        lk.unlock(); \
                        continue; \
                    } \
                } \
                cv.wait(lk); \
                lk.unlock(); \
            } \
        } \
        init(); \
    } \
} // mwsr_list
