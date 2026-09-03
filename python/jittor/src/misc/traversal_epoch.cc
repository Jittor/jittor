// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/traversal_epoch.h"

namespace jittor {

int TraversalEpoch::live_count = 0;

// The epochs that exist right now, innermost last. Nesting depth is 1 in
// almost every run and 2--3 at worst (a profiler traversal inside run_sync, a
// re-entrant run_sync inside grad), so a vector scanned linearly is the right
// shape: no allocation after the first nesting, and nothing to pay when
// nothing is nested.
static vector<TraversalEpoch*>& live_epochs() {
    static vector<TraversalEpoch*> epochs;
    return epochs;
}

TraversalEpoch::TraversalEpoch(const char* name)
    : stamp(++tflag_count), name(name) {
    live_epochs().push_back(this);
    live_count = (int)live_epochs().size();
}

TraversalEpoch::~TraversalEpoch() {
    auto& epochs = live_epochs();
    // Not necessarily the last one: an epoch may be a member of something with
    // a longer life than the traversal that made it.
    for (int i=(int)epochs.size()-1; i>=0; i--)
        if (epochs[i] == this) {
            epochs.erase(epochs.begin()+i);
            break;
        }
    live_count = (int)epochs.size();
}

// `node` is about to be re-marked. If the mark it carries belongs to an epoch
// that is still alive, that epoch has just lost a node it had visited, and
// every later answer it gives about this graph is unreliable.
void TraversalEpoch::note_overwrite(Node* node) {
    for (TraversalEpoch* epoch : live_epochs())
        if (epoch->stamp == node->tflag)
            epoch->invalidated = true;
}

void TraversalEpoch::report_invalidated() const {
    LOGf << "traversal" << name << "(epoch" << stamp >> ")"
        << "is reading marks that another traversal overwrote while it was"
        << "still walking. Its answers about which nodes it has visited are"
        << "no longer true. The two traversals have to stop sharing"
        << "Node::tflag: give the outer one its own marks (see"
        << "misc/node_index.h) or finish it before starting the inner one.";
}

} // jittor
