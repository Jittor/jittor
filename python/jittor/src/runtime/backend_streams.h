#pragma once
#include "runtime/backend.h"

namespace jittor {

void* accelerator_backend_stream(int device, BackendStreamKind kind);
void backend_side_stream_wait_default(BackendStreamKind kind, int stream_device, int default_device);
void backend_default_stream_wait_side(BackendStreamKind kind, int stream_device, int default_device);
void backend_side_stream_defer_join(BackendStreamKind kind, int device);
bool backend_side_stream_hold_block(BackendStreamKind kind, int device,
    void* ptr, size_t allocation, size_t size, Allocator* allocator);
int backend_side_stream_resolve_join(BackendStreamKind kind);
bool backend_side_stream_any_join_pending(BackendStreamKind kind);

} // namespace jittor
