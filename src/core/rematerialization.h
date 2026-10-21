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

} // namespace jittor
