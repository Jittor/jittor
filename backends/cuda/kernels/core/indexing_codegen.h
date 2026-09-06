#pragma once
#include "op.h"
#include "type/nano_vector.h"

namespace jittor {
EXTERN_LIB void cuda_loop_schedule(NanoVector shape, int* masks, int* dimensions);
EXTERN_LIB void cuda_indexing_optimize(Op* op, NanoVector shape, string& source);
} // namespace jittor
