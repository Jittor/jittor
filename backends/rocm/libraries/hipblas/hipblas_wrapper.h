#pragma once

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include "utils/log.h"

namespace jittor {

inline void check_hipblas(hipblasStatus_t status, const char* operation) {
    // A hipBLAS status is the library's outcome, not a value the caller
    // supplied; reporting it as UserError blames the wrong party. Plain
    // CHECK, like the cuTT wrapper does for cuttPlan.
    CHECK(status == HIPBLAS_STATUS_SUCCESS)
        << operation << " failed with hipBLAS status " << int(status);
}

hipblasHandle_t hipblas_bind_stream(int device);
void hipblas_shutdown() noexcept;

} // namespace jittor
