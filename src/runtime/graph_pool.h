// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions in LICENSE.txt.
#pragma once
#include "runtime/backend.h"

namespace jittor {
struct Allocation;

// Allocations outside a pool scope, CPU storage and other devices keep their
// ordinary allocator. Published inputs/outputs must be allocated outside it.
EXTERN_LIB Allocator* graph_pool_allocator(Device device);
void graph_pool_capture_complete(int64 graph, vector<Allocation>& held);
void graph_pool_release_graph(int64 graph);
int64 graph_pool_enter_launch(int64 graph);
void graph_pool_leave_launch(int64 token);
}
