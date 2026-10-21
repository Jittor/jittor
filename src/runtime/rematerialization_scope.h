// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#pragma once
#include "core/common.h"

namespace jittor {

struct Var;

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
