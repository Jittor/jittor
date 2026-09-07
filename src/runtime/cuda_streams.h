// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include "core/common.h"
#include "runtime/backend_streams.h"

namespace jittor {

struct Allocator;

enum CudaSideStreamKind {
    CUDA_COPY_STREAM = 0,
    CUDA_COMMUNICATION_STREAM = 1,
};

// @pyjt(_cuda_stream_handle)
uint64 cuda_stream_handle(int kind, int device);
// @pyjt(_cuda_stream_dependency_count)
uint64 cuda_stream_dependency_count(int kind, int device);
// Whether a join back to the default stream is still outstanding on `kind`.
// Exposed so a test can tell "the join was deferred" apart from "the join
// happened and the overlap window was zero" -- the two are indistinguishable
// from the outside otherwise.
// @pyjt(_cuda_stream_join_pending)
bool cuda_stream_join_pending(int kind, int device);

inline BackendStreamKind backend_side_kind(CudaSideStreamKind kind) {
    CHECK(kind == CUDA_COPY_STREAM || kind == CUDA_COMMUNICATION_STREAM)
        << "Invalid side-stream kind";
    return kind == CUDA_COPY_STREAM ? BackendStreamKind::Copy : BackendStreamKind::Communication;
}
inline void cuda_side_stream_wait_default(CudaSideStreamKind kind, int stream_device, int default_device) {
    backend_side_stream_wait_default(backend_side_kind(kind), stream_device, default_device);
}
inline void cuda_default_stream_wait_side(CudaSideStreamKind kind, int stream_device, int default_device) {
    backend_default_stream_wait_side(backend_side_kind(kind), stream_device, default_device);
}

/**
Record the side stream's done event but leave the default stream free to run
ahead of it.

This is the half of the stream contract that makes overlap possible at all:
`cuda_default_stream_wait_side` orders the default stream behind the side
stream *at the point it is called*, so calling it right after the work is
enqueued means nothing can ever overlap with that work. Deferring the join
moves the ordering to wherever the result is actually needed.

The caller owes two things in exchange:

* every block the side stream still reads or writes must be handed to
  `cuda_side_stream_hold_block` first, because the default stream running
  ahead includes the allocator handing those blocks out again, and
* `cuda_side_stream_resolve_join` must be called before the results are read.
*/
inline void cuda_side_stream_defer_join(CudaSideStreamKind kind, int device) {
    backend_side_stream_defer_join(backend_side_kind(kind), device);
}

// Keep one block reserved until the pending join resolves. The caller must
// have taken the extra reference already (`allocator->share_with`); this takes
// ownership of that reference. Returns false if the allocator cannot share, in
// which case the caller must not defer the join.
inline bool cuda_side_stream_hold_block(
    CudaSideStreamKind kind, int device,
    void* ptr, size_t allocation, size_t size, Allocator* allocator) {
    return backend_side_stream_hold_block(backend_side_kind(kind), device, ptr, allocation, size, allocator);
}

// Order the default stream behind the side stream on every device with a
// pending join, and release the blocks those joins were holding. Releasing is
// safe only here: once the default stream waits on the done event, a block
// handed out again can only reach work that is already ordered after the side
// stream. Returns the number of devices joined.
inline int cuda_side_stream_resolve_join(CudaSideStreamKind kind) {
    return backend_side_stream_resolve_join(backend_side_kind(kind));
}

// Whether any device still owes a join on `kind`.
inline bool cuda_side_stream_any_join_pending(CudaSideStreamKind kind) {
    return backend_side_stream_any_join_pending(backend_side_kind(kind));
}

} // namespace jittor
