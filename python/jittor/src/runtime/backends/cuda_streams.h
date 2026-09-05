#pragma once
#include "runtime/backend.h"

namespace jittor {
#ifdef HAS_CUDA
void* accelerator_backend_stream(int device, BackendStreamKind kind);
#endif
} // namespace jittor
