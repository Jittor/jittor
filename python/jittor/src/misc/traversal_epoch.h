// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "node.h"

namespace jittor {

/**
 * One traversal's claim on `Node::tflag`.
 *
 * A stamp from `tflag_count` keeps the common membership test to one integer
 * comparison. The object owns that stamp and, unlike the old bare integer,
 * also owns the work needed to preserve it across nesting.
 *
 * That is not hypothetical. `MemoryProfiler::check()` runs a full `bfs_both`
 * from inside `Executor::run_sync`'s op loop; building a backward op re-enters
 * `run_sync` through `Op::init`. Both are legal, so the inner traversal must
 * return the outer traversal's marks rather than relying on call order.
 *
 * This makes the claim an object and makes synchronous nesting reversible:
 *
 *   - an epoch that runs inside another records the stamp it displaced;
 *   - its destructor restores those stamps in reverse order before the outer
 *     traversal resumes;
 *   - the common, non-nested path still writes only `Node::tflag`, without a
 *     side allocation or hash lookup.
 *
 * Traversals are synchronous and stack-nested. They must not destroy a node
 * they marked before their epoch ends; graph walks already require their nodes
 * to stay alive for the duration of the walk.
 */
struct TraversalEpoch {
    int64 stamp;
    const char* name;
    vector<pair<Node*, int64>> displaced;

    explicit TraversalEpoch(const char* name);
    ~TraversalEpoch();

    TraversalEpoch(const TraversalEpoch&) = delete;
    TraversalEpoch& operator=(const TraversalEpoch&) = delete;

    /// Claim `node` for this traversal.
    inline void mark(Node* node) {
        if (node->tflag == stamp) return;
        // Only a nested traversal needs a restoration log. The branch is cold
        // in ordinary execution, where live_count is exactly one.
        if (PREDICT_BRANCH_NOT_TAKEN(live_count > 1))
            displaced.emplace_back(node, node->tflag);
        node->tflag = stamp;
    }

    /// Whether `node` was claimed by this traversal.
    inline bool marked(const Node* node) const {
        return node->tflag == stamp;
    }

    /// Number of epochs alive right now; 1 is the allocation-free fast path.
    static int live_count;
};

} // jittor
