#pragma once

#include "core/common.h"

namespace jittor {

struct VarHolder;

// Finish a refused trace without repeating work which already ran kept.
// @pyjt(_finish_step_readouts)
void finish_step_readouts(const vector<VarHolder*>& roots);

} // namespace jittor
