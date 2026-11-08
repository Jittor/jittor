// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#pragma once
#include "core/common.h"

namespace jittor {

struct Var;

// Construction provenance belongs to the calling thread, not a global flag.
EXTERN_LIB uint64 rematerialization_construction_owner();
EXTERN_LIB bool rematerialization_has_recording_owner();
EXTERN_LIB bool rematerialization_preserves_rng();
EXTERN_LIB void rematerialization_mark_kept(Var* value);

// Must be nested inside ExecutorEntryScope and span the complete submission.
struct RematerializationExecutionScope {
    void* frame = nullptr;
    RematerializationExecutionScope();
    ~RematerializationExecutionScope();
    RematerializationExecutionScope(const RematerializationExecutionScope&) = delete;
    RematerializationExecutionScope& operator=(const RematerializationExecutionScope&) = delete;
};

// Used under executor entry ownership. Existing retained buffers belong to
// their caller and must not become disposable keep_graph=2 intermediates.
struct RematerializationRetentionScope {
    int previous_keep_graph;
    void* frame = nullptr;
    explicit RematerializationRetentionScope(const vector<Var*>& roots);
    ~RematerializationRetentionScope();
    RematerializationRetentionScope(const RematerializationRetentionScope&) = delete;
    RematerializationRetentionScope& operator=(const RematerializationRetentionScope&) = delete;
};

// Queued callbacks are separate user work, including when they reenter the
// executor on the materializing thread. Explicit nested materializers still
// establish their own retention policy.
struct RematerializationCallbackScope {
    void* owner_callback = nullptr;
    bool active = false;
    int previous_keep_graph = 0;
    size_t previous_visible_from = 0;
    RematerializationCallbackScope();
    ~RematerializationCallbackScope();
    RematerializationCallbackScope(const RematerializationCallbackScope&) = delete;
    RematerializationCallbackScope& operator=(const RematerializationCallbackScope&) = delete;
};

EXTERN_LIB bool rematerialization_allows_release(Var* value);

} // namespace jittor
