#pragma once
#include "core/common.h"

namespace jittor {

struct Var;

struct DispatchContext {
    string backend;
    int device_id;
};

// Query the next operation's placement, not the operands' current storage.
// Pending constants may follow another input, but this query never retargets
// their graph. The eventual Op::propagate_device performs that mutation.
DispatchContext query_dispatch_context(const vector<Var*>& inputs);

} // namespace jittor
