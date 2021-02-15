// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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

struct EventQueue {
    static constexpr int RUNNING = 0;
    static constexpr int OK = 1;
    static constexpr int ERROR = 2;
    typedef void(*Func)();
    struct Worker {
        Func todo;
        std::condition_variable cv;
        std::mutex mtx;
        std::thread thread = std::thread(Worker::start);

        static void start();

        inline void run(Func func) {
            {
                std::lock_guard<std::mutex> l(mtx);
                todo = func;
            }
            cv.notify_one();
        }

        inline ~Worker() {
            run(nullptr);
            thread.join();
        }
    } worker;

    list<Func> tasks;
    std::condition_variable cv;
    std::mutex mtx;
    Func func;
    volatile int run_sync_done;

    inline void flush() {
        list<Func> ts;
        {
            std::lock_guard<std::mutex> g(mtx);
            ts = move(tasks);
        }
        for (auto func : ts)
            func();
    }

    static void worker_caller();

    int run_sync(Func func) {
        // send work to worker and do something by self
        std::unique_lock<std::mutex> l(mtx);
        this->func = func;
        run_sync_done = RUNNING;
        // send func to worker
        worker.run(worker_caller);
        while (1) {
            // check self work or worker's status
            cv.wait(l);
            list<Func> ts = move(tasks);
            l.unlock();
            // do self works
            for (auto func : ts)
                func();
            l.lock();
            // worker is finished
            if (int ret = run_sync_done)
                return ret;
        }
    }

    inline void push(Func func) {
        {
            std::lock_guard<std::mutex> g(mtx);
            tasks.push_back(func);
        }
        cv.notify_one();
    }
};

extern EventQueue event_queue;

} // jittor