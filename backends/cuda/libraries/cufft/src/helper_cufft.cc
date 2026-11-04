// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// checkCudaErrors() on a cufftResult needs this overload. The copy in
// The common helper_cuda.cc is compiled without <cufft.h> in scope, so
// its #ifdef _CUFFT_H_ body never reaches libcuda_extern; every other backend
// carries its own helper_*.cc for the same reason.

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "utils/log.h"
#include "helper_cuda.h"

const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
#if CUFFT_VERSION < 12000
    // cuFFT 12 removed these obsolete result enumerators from its headers.
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
#if CUFFT_VERSION < 12000
    case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
#endif
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
#if CUFFT_VERSION < 12000
    case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
#endif
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
  }
  return "<unknown>";
}
