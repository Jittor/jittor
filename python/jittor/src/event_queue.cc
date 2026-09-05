// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
            self->cv.wait(l);
            todo = self->todo;
        }
        if (!todo) break;
        todo();
    }
}


// The worker can be stopped from two places, in either order, and each of
// them can be reached twice:
//
//  - `cleanup_callback`, drained by core.cleanup() (python's atexit) and by
//    log_exiting() (a C-level std::atexit registered in log.cc);
//  - ~Worker, during static destruction of the global `event_queue`.
//
// Neither knows about the other. `stop()` is a static function that reaches
// the global, so running it after ~Worker is a use-after-destruction, and
// join() on the already-joined thread throws std::system_error(EINVAL) out of
// an atexit handler -- std::terminate, from a process that was exiting
// cleanly. One flag makes every order safe and every call after the first a
// no-op.
static bool worker_stopped = false;

void EventQueue::Worker::stop() {
    if (worker_stopped) return;
    worker_stopped = true;
    if (!event_queue.worker.thread.joinable()) return;
    event_queue.worker.run(nullptr);
    event_queue.worker.thread.join();
}

EXTERN_LIB vector<void(*)()> cleanup_callback;

EventQueue::Worker::Worker() : thread(EventQueue::Worker::start) {
    cleanup_callback.push_back(&EventQueue::Worker::stop);
}

EventQueue::Worker::~Worker() {
    // Runs during static destruction of the global `event_queue`. Normally the
    // worker is already stopped through cleanup_callback -- but that list is
    // drained by core.cleanup(), whose atexit registration sits at the end of
    // the python module that `import jittor` was still executing when it
    // raised. A failed import therefore reached ~std::thread with a joinable
    // thread and called std::terminate: "terminate called without an active
    // exception", SIGABRT, and (through the SIGCHLD handler in log.cc) a
    // parent process that saw nothing at all.
    stop();
}

#endif


} // jittor
