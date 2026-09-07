// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

// Entry into the executor, and the device waits inside it.
//
// Why the two are one header: the executor used to hold the GIL from the
// moment it was entered until it returned, and that -- not any lock -- is what
// made `Executor::run_sync`'s "not reentrant" contract (see executor.h) hold
// for Python threads. Handing the GIL to another thread during the device wait
// removes that guarantee, so the guarantee has to be restated as a lock before
// the GIL can be dropped anywhere inside. One without the other trades a
// serialization point for a data race.
//
// ExecutorEntryScope is the lock. It is recursive *by thread*: a nested batch
// (dynamic shape inference re-enters `run_sync` from `Op::init`, and building
// backward ops re-enters it from `grad`) sees a depth > 0 and neither locks nor
// touches the GIL, so nesting costs nothing and cannot self-deadlock.
//
// The acquisition order is the part that matters. Taking the lock while
// holding the GIL inverts against the release inside:
//
//     A: holds the lock, dropped the GIL, wants the GIL back to return
//     B: holds the GIL, waits for the lock
//     -> B never drops the GIL, A never gets it back, both wait forever.
//
// So the scope drops the GIL *first*, then blocks on the lock, then takes the
// GIL back with the lock held. A thread waiting for the executor therefore
// never holds the GIL, and the inversion has nowhere to form.
//
// The executor is also reachable from threads that hold no GIL and have no
// Python thread state at all (a device host callback, the event queue worker).
// Those must still take the lock; they simply skip the GIL half, which
// `GILReleaseScope` decides for itself by asking PyGILState_Check().
struct ExecutorEntryScope {
    // Whether this scope is the one that took the lock, as opposed to a nested
    // entry on a thread that already holds it.
    bool owns;
    ExecutorEntryScope();
    ~ExecutorEntryScope();
    ExecutorEntryScope(const ExecutorEntryScope&) = delete;
    ExecutorEntryScope& operator=(const ExecutorEntryScope&) = delete;
};

// Whether this thread is inside an ExecutorEntryScope, i.e. whether it holds
// the executor lock. Equivalently: whether dropping the GIL right now is safe.
bool inside_executor();

// RAII for a segment that only waits on the device: a full-device
// synchronization, or a blocking host/device copy. Releases the GIL so another
// Python thread can run, and takes it back on the way out (including exception
// unwind, so a throwing wait reaches its handler with the GIL held again).
//
// Does nothing unless this thread holds the executor lock. That is not an
// optimization -- it is what keeps the release safe. The lock is the reason no
// other thread can be inside the executor during the window, so a device wait
// reached without it (there is none today; the check is a guard against the
// next caller) keeps the GIL and stays serialized the old way.
//
// What must *not* go inside such a segment:
//
//   * `event_queue.flush()`, which runs fetch callbacks that touch Python
//     objects. It sits after the phase 7 wait, deliberately outside.
//   * the error path. `check_op_async_error` reads `TraceData` through
//     `get_node_trace`, i.e. Python frame objects recorded by the tracer. The
//     scope's destructor runs before any catch handler in an enclosing try, so
//     a wait that throws is already back under the GIL when that runs.
struct DeviceWaitScope {
    // PyThreadState*, kept opaque so core headers stay free of Python.h.
    void* saved;
    DeviceWaitScope();
    ~DeviceWaitScope();
    DeviceWaitScope(const DeviceWaitScope&) = delete;
    DeviceWaitScope& operator=(const DeviceWaitScope&) = delete;
};

} // jittor
