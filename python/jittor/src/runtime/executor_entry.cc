// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "pyjt/gil.h"
#include <mutex>
#include "runtime/executor_entry.h"

namespace jittor {

// One batch at a time, process-wide. std::mutex is trivially destructible
// under libstdc++, so this cannot become a use-after-destruction for a thread
// that is still shutting down (the event queue worker has that history, see
// event_queue.cc).
static std::mutex entry_mutex;

// Nesting depth on *this* thread. depth > 0 is equivalent to "this thread
// holds entry_mutex", which is why it is only raised once the lock is really
// held and lowered before it is released.
static thread_local int entry_depth = 0;

bool inside_executor() { return entry_depth > 0; }

ExecutorEntryScope::ExecutorEntryScope() : owns(entry_depth == 0) {
    if (owns) {
        // Drop the GIL before blocking, take it back with the lock held; see
        // the inversion in the header.
        GILReleaseScope gil_release;
        entry_mutex.lock();
    }
    entry_depth++;
}

ExecutorEntryScope::~ExecutorEntryScope() {
    entry_depth--;
    if (owns) entry_mutex.unlock();
}

DeviceWaitScope::DeviceWaitScope() : saved(nullptr) {
    if (!inside_executor()) return;
    if (Py_IsInitialized() && PyGILState_Check())
        saved = (void*)PyEval_SaveThread();
}

DeviceWaitScope::~DeviceWaitScope() {
    if (saved) PyEval_RestoreThread((PyThreadState*)saved);
}

} // jittor
