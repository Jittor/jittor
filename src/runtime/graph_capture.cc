// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/graph_capture.h"
#include "runtime/backend.h"
#include "runtime/async_executor.h"

namespace jittor {

bool graph_capture_supported() { return backend_graph_supported(); }

// A recording captures the calling thread's stream, so nothing the
// asynchronous executor still has queued may land in it or run beside it.
bool graph_capture_begin() {
    async_drain();
    return backend_graph_capture_begin();
}

int64 graph_capture_end() {
    return reinterpret_cast<int64>(backend_graph_capture_end());
}

void graph_launch(int64 graph) {
    async_drain();
    backend_graph_launch(reinterpret_cast<void*>(graph));
}

void graph_release(int64 graph) {
    backend_graph_release(reinterpret_cast<void*>(graph));
}

} // jittor
