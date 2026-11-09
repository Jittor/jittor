#pragma once

#include <omp.h>
#include <stdexcept>

namespace jittor {

// @pyjt(get_num_threads)
inline int get_num_threads() {
    return omp_get_max_threads();
}

// @pyjt(set_num_threads)
inline void set_num_threads(int num_threads) {
    if (num_threads < 1) {
        throw std::invalid_argument("num_threads must be positive");
    }
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
}

} // namespace jittor
