// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>

namespace jittor {

// RAII: release the Python GIL for the duration of the scope and reacquire it
// on the way out, including exception unwind.
//
// Two callers, for two different reasons:
//
//   * the parallel op compiler (`parallel_compiler.cc`). The main thread gets
//     here from a pybind call and therefore holds the GIL, and then waits on
//     the compile worker futures. Those workers may call `py_caller()` (the
//     `@python` JIT pass), which takes the GIL via PyGILState_Ensure. If the
//     main thread kept the GIL during its wait, the workers could never take
//     it -> deadlock.
//   * the executor's device-wait segments (`runtime/executor_entry.h`), where
//     the point is the opposite one: nothing on this thread needs the GIL
//     while it blocks on the device, so another Python thread may as well run.
//
// Both guards below are load-bearing:
//
//   Py_IsInitialized()  a pure C++ embedding of the core has no interpreter at
//                       all, and the CPython calls would fault.
//   PyGILState_Check()  PyEval_SaveThread() on a thread that does *not* hold
//                       the GIL is fatal ("drop the GIL: ..." -> abort), not a
//                       no-op. The parallel compiler's caller always holds the
//                       GIL, so that path never reached it; the executor is
//                       reachable from threads that hold no GIL and never had
//                       a thread state (a device callback, the event queue
//                       worker), so the check has to be here rather than at
//                       each call site.
struct GILReleaseScope {
    PyThreadState* save = nullptr;
    inline GILReleaseScope() {
        if (Py_IsInitialized() && PyGILState_Check())
            save = PyEval_SaveThread();
    }
    inline ~GILReleaseScope() {
        if (save)
            PyEval_RestoreThread(save);
    }
    GILReleaseScope(const GILReleaseScope&) = delete;
    GILReleaseScope& operator=(const GILReleaseScope&) = delete;
};

} // jittor
