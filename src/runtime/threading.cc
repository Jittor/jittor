#include "runtime/threading.h"
#include <omp.h>

namespace jittor {

int runtime_openmp_max_threads() {
    return omp_get_max_threads();
}

} // namespace jittor
