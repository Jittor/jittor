// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <iostream>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace jittor {
    
struct AsyncQueue {
    AsyncQueue() : stop(false), prevTaskCompleted(true) {
        worker = std::thread([this]() { this->workerThread(); });
    }

    ~AsyncQueue() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
            condition.notify_all();
        }
        worker.join();
    }

    template <typename F, typename... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            tasks.push(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }
    
    void waitAllTasksComplete() {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]() { return tasks.empty() && prevTaskCompleted; });
    }

private:
    void workerThread() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [this]() { return stop || (!tasks.empty() && prevTaskCompleted); });
                if (stop && tasks.empty()) {
                    return;
                }
                prevTaskCompleted = false;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lock(mutex);
                prevTaskCompleted = true;
            }
            condition.notify_one(); // 完成一个任务后通知等待的线程
        }
    }

    std::queue<std::function<void()>> tasks;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable condition;
    bool stop;
    bool prevTaskCompleted;
};

} // jittor
