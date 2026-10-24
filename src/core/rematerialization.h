// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#pragma once
#include "core/var_holder.h"

namespace jittor {

// Internal building block, not a complete checkpoint implementation. Execute
// the requested graph once while retaining its topology, then return detached
// storage owners whose reads do not execute that retained graph again.
// @pyjt(_materialize_remat_outputs)
vector<VarHolder*> materialize_remat_outputs(const vector<VarHolder*>& outputs);

struct RematerializationOwnerFrame;

// Private construction recording and exact native topology ownership. Closing
// stops recording; release gives back only this lease's introduced kept bits.
// @pyjt(_RematerializationLease)
// @attrs(heaptype)
struct RematerializationLease {
    std::shared_ptr<RematerializationOwnerFrame> frame;
    // @pyjt(__init__)
    RematerializationLease(bool preserve_rng_state=true);
    // @pyjt(__dealloc__)
    ~RematerializationLease();
    // @pyjt(close)
    void close();
    // @pyjt(release)
    void release();
    // @pyjt(pin_count)
    int64 pin_count();
};

} // namespace jittor
