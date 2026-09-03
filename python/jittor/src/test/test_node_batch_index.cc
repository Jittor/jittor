// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "node.h"
#include "misc/node_index.h"

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

    int64 mine = ++tflag_count;
    node->set_batch_index(mine, 7);
    CHECKop(node->batch_index_at(mine),==,7);

    // A second traversal renumbers the same node. This is exactly what the
    // shared custom_data used to allow silently: the first traversal kept
    // reading, and got the second one's index.
    int64 theirs = ++tflag_count;
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

} // jittor
