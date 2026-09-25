// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/graph_capture.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include <unordered_map>

namespace jittor {

bool graph_capture_supported() { return backend_graph_supported(); }

// The frees each live recording holds, by handle; see `capture_held_frees`.
// Destroying an entry performs them.
static std::unordered_map<int64, vector<Allocation>> held_by_graph;

bool graph_capture_begin() {
    if (!backend_graph_capture_begin()) return false;
    begin_capture_hold();
    return true;
}

int64 graph_capture_end() {
    int64 graph = reinterpret_cast<int64>(backend_graph_capture_end());
    // Out of the pools' reach before anything is freed; without a graph to
    // own them, the held blocks are freed right here.
    auto held = end_capture_hold();
    if (graph) held_by_graph[graph] = std::move(held);
    return graph;
}

void graph_launch(int64 graph) {
    backend_graph_launch(reinterpret_cast<void*>(graph));
}

void graph_wait() {
    if (backend_graph_supported())
        backend_synchronize({accelerator_backend_id(), current_device()});
}

void graph_release(int64 graph) {
    backend_graph_release(reinterpret_cast<void*>(graph));
    held_by_graph.erase(graph);
}

} // jittor
