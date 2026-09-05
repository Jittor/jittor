// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include "common.h"

namespace jittor {

#ifdef HAS_CUDA
struct EventQueue {
    typedef void(*Func)();

    list<Func> tasks;
    std::mutex mtx;

    struct Worker {
        Func todo;
        std::condition_variable cv;
        std::mutex mtx;
        std::thread thread;

        static void start();
        static void stop();

        Worker();
        ~Worker();

        inline void run(Func func) {
            {
                std::lock_guard<std::mutex> l(mtx);
                todo = func;
            }
            cv.notify_one();
        }
    } worker;

    inline void flush() {
        list<Func> ts;
        {
            std::lock_guard<std::mutex> g(mtx);
            ts = move(tasks);
        }
        for (auto func : ts)
            func();
    }

    inline void push(Func func) {
        {
            std::lock_guard<std::mutex> g(mtx);
            tasks.push_back(func);
        }
    }
};

EXTERN_LIB EventQueue event_queue;

#endif

} // jittor
