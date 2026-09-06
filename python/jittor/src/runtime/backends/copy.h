#pragma once
#include "runtime/backend.h"

namespace jittor {

void cpu_backend_copy(void* dst, Device dst_device, const void* src,
                      Device src_device, size_t size, bool ordered);
void cpu_backend_copy_async(void* dst, Device dst_device, const void* src,
                            Device src_device, size_t size, BackendStream stream);

} // namespace jittor
