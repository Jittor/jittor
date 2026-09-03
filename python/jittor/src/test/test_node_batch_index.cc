// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "node.h"
#include "misc/node_index.h"
#include "misc/traversal_epoch.h"

namespace jittor {

// Node::batch_index is the one per-node scratch slot left after 2.02. It is
// allowed to stay on the node only because reading it is *checked*: the reader
// names the batch it believes it is in, and a node stamped by a different
// traversal answers with an error instead of that traversal's number.
//
// This is the mechanism, tested directly. The behavioural side --
// MemoryProfiler::check() running a full traversal from inside run_sync's op
// loop and the executor's numbering surviving it -- is
// tests/core/test_traversal_state_isolation.py.
JIT_TEST(node_batch_index_is_checked) {
    VarPtr a({4}, "float32");
    Node* node = a.ptr;

    TraversalEpoch mine_epoch("batch_index_mine");
    int64 mine = mine_epoch.stamp;
    node->set_batch_index(mine, 7);
    CHECKop(node->batch_index_at(mine),==,7);

    // A second traversal renumbers the same node. This is exactly what the
    // shared custom_data used to allow silently: the first traversal kept
    // reading, and got the second one's index.
    TraversalEpoch their_epoch("batch_index_theirs");
    int64 theirs = their_epoch.stamp;
    node->set_batch_index(theirs, 0);
    CHECKop(node->batch_index_at(theirs),==,0);
    expect_error([&]() { node->batch_index_at(mine); });

    // A node no traversal has stamped is not silently "index 0" either.
    VarPtr b({4}, "float32");
    expect_error([&]() { b.ptr->batch_index_at(mine); });
}

// NodeIndex is what the traversals that gave the slot up use instead. The two
// properties their callers depend on: a reference handed out stays valid (the
// topological sorts do `--index[node]`), and a node that was never indexed is
// distinguishable from one indexed as 0.
JIT_TEST(node_index_table) {
    vector<VarPtr> vars;
    for (int i=0; i<64; i++) vars.emplace_back(NanoVector(4), "float32");

    NodeIndex index;
    index.reset(vars.size());
    for (uint i=0; i<vars.size(); i++) index[vars[i].ptr] = i;
    for (uint i=0; i<vars.size(); i++) {
        CHECKop(index.get(vars[i].ptr),==,(int)i);
        CHECK(index.has(vars[i].ptr));
    }

    VarPtr absent({4}, "float32");
    CHECK(!index.has(absent.ptr));
    CHECKop(index.get(absent.ptr, -1),==,-1);

    // References stay put: no rehash can move them, because the capacity was
    // fixed by reset().
    int& slot = index[vars[3].ptr];
    for (uint i=0; i<vars.size(); i++) index[vars[i].ptr] += 1000;
    CHECKop(slot,==,1003);
    slot--;
    CHECKop(index.get(vars[3].ptr),==,1002);

    // reset() starts a new generation: everything is absent again, and the
    // table is reused rather than reallocated.
    index.reset(vars.size());
    for (uint i=0; i<vars.size(); i++) CHECK(!index.has(vars[i].ptr));
}



// A nested synchronous traversal may borrow Node::tflag, but it has to return
// every outer mark before the outer traversal resumes. Keeping the restoration
// log on the inner epoch pays nothing on the common, non-nested path.
JIT_TEST(traversal_epoch_restores_outer_marks) {
    VarPtr a({4}, "float32");
    VarPtr b({4}, "float32");
    {
        TraversalEpoch outer("outer");
        outer.mark(a.ptr);
        CHECK(outer.marked(a.ptr));
        CHECK(!outer.marked(b.ptr));
        {
            TraversalEpoch inner("inner");
            // A nested traversal over nodes the outer one never reached is
            // not a problem, and must not be reported as one.
            inner.mark(b.ptr);
            CHECK(outer.marked(a.ptr));
            // This temporarily takes a node the outer traversal had visited.
            inner.mark(a.ptr);
            CHECK(inner.marked(a.ptr));
        }
        // Synchronous nesting has returned, so the outer walk must be able to
        // continue without revisiting a or losing b's pre-existing stamp.
        CHECK(outer.marked(a.ptr));
        CHECK(!outer.marked(b.ptr));
    }
    // A later traversal is unaffected by any of that.
    TraversalEpoch solo("solo");
    solo.mark(a.ptr);
    CHECK(solo.marked(a.ptr));
    CHECK(!solo.marked(b.ptr));
}

} // jittor
