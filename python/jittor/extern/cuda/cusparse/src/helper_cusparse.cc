/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking
#include <cuda_runtime.h>
#include <cusparse.h>
#include "helper_cuda.h"

#ifdef CUSPARSEAPI
// cuSPARSE API errors
const char *_cudaGetErrorEnum(cusparseStatus_t status) {
  switch(status)
  {
    case CUSPARSE_STATUS_SUCCESS:
      return "success";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      return "unknown error";
  }
}
#endif