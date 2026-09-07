#pragma once
#include "runtime/dispatch_context.h"
#include "core/var_holder.h"

namespace jittor {

// @pyjt(dispatch_context)
inline DispatchContext dispatch_context(const vector<VarHolder*>& inputs) {
    return query_dispatch_context(convert(inputs));
}

} // namespace jittor
