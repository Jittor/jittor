// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "core/node.h"
#include "runtime/traversal_state.h"

namespace jittor {

/**
 * One traversal's claim on `Node::tflag`.
 *
 * A stamp from the runtime traversal state keeps membership to one integer
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
private:
    RuntimeTraversalState& state_;

public:
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
        // in ordinary execution, where one epoch is active.
        if (PREDICT_BRANCH_NOT_TAKEN(state_.active_epochs() > 1))
            displaced.emplace_back(node, node->tflag);
        node->tflag = stamp;
    }

    /// Whether `node` was claimed by this traversal.
    inline bool marked(const Node* node) const {
        return node->tflag == stamp;
    }

};

// A queue walk needs two states, not two nested traversals. Reserving the
// expansion stamp in a short empty epoch leaves this owner as the only active
// walk. Ordinary queue/expanded transitions therefore allocate no restoration
// log; true nesting still restores whichever outer state was borrowed.
struct TraversalQueueMarks final : TraversalEpoch {
    int64 expanded_stamp;
    static int64 reserve_expansion_stamp() {
        TraversalEpoch reservation("queue expansion stamp");
        return reservation.stamp;
    }
    explicit TraversalQueueMarks(const char* name)
        : TraversalEpoch(name), expanded_stamp(reserve_expansion_stamp()) {}
    bool contains(const Node* node) const {
        return node->tflag == stamp || node->tflag == expanded_stamp;
    }
    bool expanded(const Node* node) const { return node->tflag == expanded_stamp; }
    void mark_queued(Node* node) {
        if (!contains(node)) TraversalEpoch::mark(node);
    }
    void mark_expanded(Node* node) {
        mark_queued(node);
        node->tflag = expanded_stamp;
    }
    ~TraversalQueueMarks() {
        // The base destructor restores its one stamp. Convert only marks this
        // owner still holds; a nested owner has already returned its marks.
        for (const auto& item : displaced)
            if (item.first->tflag == expanded_stamp) item.first->tflag = stamp;
    }
};

} // jittor
