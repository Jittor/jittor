// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/traversal_epoch.h"

namespace jittor {

TraversalEpoch::TraversalEpoch(const char* name)
    : state_(runtime_traversal_state()), stamp(state_.enter()), name(name) {}

TraversalEpoch::~TraversalEpoch() {
    for (auto i=displaced.rbegin(); i!=displaced.rend(); ++i)
        if (i->first->tflag == stamp)
            i->first->tflag = i->second;
    state_.leave();
}

} // jittor
