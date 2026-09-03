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
 * Marking visited nodes is done by writing a number into `Node::tflag` and
 * comparing against it, where the number comes from a single global counter
 * (`tflag_count`). That works for one traversal at a time and has no way to
 * say so: a traversal that starts while another is still walking takes a new
 * number and **overwrites the marks of the outer one**, which then sees its
 * own nodes as unvisited and quietly walks a graph it has already walked, or
 * skips one it has not.
 *
 * That is not hypothetical. `MemoryProfiler::check()` runs a full `bfs_both`
 * from inside `Executor::run_sync`'s op loop; building a backward op re-enters
 * `run_sync` through `Op::init`. Both are legal and neither is announced. The
 * code survives it by *not looking* -- run_sync finishes every tflag question
 * before the loop starts, and `grad()` copies the indices it will need into a
 * side buffer first -- so the invariant is "whoever reads tflag must already
 * have stopped reading it", held together by call order.
 *
 * This makes the claim an object, and makes losing it *detectable* rather than
 * silent:
 *
 *   - a live epoch is registered while it exists;
 *   - `mark()` notices when it is overwriting a mark that a still-live outer
 *     epoch owns, and invalidates that epoch -- precisely, so an inner
 *     traversal over untouched nodes costs the outer one nothing;
 *   - reading an invalidated epoch is an error naming both traversals,
 *     instead of an answer that is wrong for reasons three frames away.
 *
 * It deliberately does *not* try to let two traversals mark the same node at
 * once. That needs per-traversal storage, and per-traversal storage on this
 * path was measured at +43% of run_sync (see
 * agent/skills/jittor-core-planning-cost). Detection is what turns the
 * convention into a contract; giving up the sharing is a separate question.
 */
struct TraversalEpoch {
    int64 stamp;
    const char* name;
    bool invalidated = false;

    explicit TraversalEpoch(const char* name);
    ~TraversalEpoch();

    TraversalEpoch(const TraversalEpoch&) = delete;
    TraversalEpoch& operator=(const TraversalEpoch&) = delete;

    /// Claim `node` for this traversal.
    inline void mark(Node* node) {
        // Cheap test first: only a node that already carries *some* live
        // epoch's mark can be taking one away, and nesting is rare.
        if (PREDICT_BRANCH_NOT_TAKEN(node->tflag != stamp && live_count > 1))
            note_overwrite(node);
        node->tflag = stamp;
    }

    /// Whether `node` was claimed by this traversal.
    inline bool marked(const Node* node) const {
        if (PREDICT_BRANCH_NOT_TAKEN(invalidated)) report_invalidated();
        return node->tflag == stamp;
    }

    void report_invalidated() const;
    static void note_overwrite(Node* node);
    /// Number of epochs alive right now; 1 means nothing can be overwritten.
    static int live_count;
};

} // jittor
