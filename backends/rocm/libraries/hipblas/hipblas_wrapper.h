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
void hipblas_sgemm_policy(int device, hipblasOperation_t trans_a, hipblasOperation_t trans_b,
    int m, int n, int k, const float* a, int lda, const float* b, int ldb,
    float* c, int ldc, bool inference, int64 stride_a=0, int64 stride_b=0,
    int64 stride_c=0, int batch_count=0);
void hipblas_require_fp64(int device);
void hipblas_shutdown() noexcept;

} // namespace jittor
