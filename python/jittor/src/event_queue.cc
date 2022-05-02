// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "event_queue.h"

namespace jittor {

#ifdef HAS_CUDA
EventQueue event_queue;

void EventQueue::Worker::start() {
    Worker* self = &event_queue.worker;
    while (1) {
        Func todo;
        {
            std::unique_lock<std::mutex> l(self->mtx);
            event_queue.cv.notify_one();
            self->cv.wait(l);
            todo = self->todo;
        }
        if (!todo) break;
        todo();
    }
}


void EventQueue::Worker::stop() {
    LOGv << "stoping event queue worker...";
    event_queue.worker.run(nullptr);
    event_queue.worker.thread.join();
    LOGv << "stopped event queue worker.";
}

EXTERN_LIB vector<void(*)()> cleanup_callback;

EventQueue::Worker::Worker() : thread(EventQueue::Worker::start) {
    cleanup_callback.push_back(&EventQueue::Worker::stop);
}

void EventQueue::worker_caller() {
    int status = OK;
    try {
        event_queue.func();
    } catch (const std::exception& e) {
        LOGe << "Catch error:\n" >> e.what();
        status = ERROR;
    }
    {
        std::lock_guard<std::mutex> l(event_queue.mtx);
        event_queue.run_sync_done = status;
    }
}

#endif


} // jittor