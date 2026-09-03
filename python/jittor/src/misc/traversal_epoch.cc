// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/traversal_epoch.h"

namespace jittor {

int TraversalEpoch::live_count = 0;

TraversalEpoch::TraversalEpoch(const char* name)
    : stamp(++tflag_count), name(name) {
    live_count++;
}

TraversalEpoch::~TraversalEpoch() {
    for (auto i=displaced.rbegin(); i!=displaced.rend(); ++i)
        if (i->first->tflag == stamp)
            i->first->tflag = i->second;
    live_count--;
}

} // jittor
