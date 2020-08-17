// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "event_queue.h"

namespace jittor {

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


} // jittor