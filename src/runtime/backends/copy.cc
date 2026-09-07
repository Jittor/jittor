#include "runtime/backends/copy.h"
#include <cstring>

namespace jittor {
void cpu_backend_copy(void* dst, Device target, const void* src, Device source, size_t size, bool) {
    CHECK(target.backend == BackendId::Cpu && source.backend == BackendId::Cpu)
        << "CPU copy requires host memory";
    if (size) std::memcpy(dst, src, size);
}
void cpu_backend_copy_async(void* dst, Device target, const void* src,
                            Device source, size_t size, BackendStream) {
    cpu_backend_copy(dst, target, src, source, size, false);
}
} // namespace jittor
