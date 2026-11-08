// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "utils/log.h"
#include "helper_cuda.h"
#include "core/common.h"
#include "type/nano_string.h"

namespace jittor {

// One dense cuSOLVER handle per device, as cublas_wrapper.cc keeps cuBLAS's;
// the global is the current device's.
EXTERN_LIB cusolverDnHandle_t cusolver_handle;
// The handle, on Jittor's compute stream of the current device.
cusolverDnHandle_t cusolver_bind_stream();
// @pyjt(cusolver_stream_bind_count)
uint64 cusolver_stream_bind_count(int device);

// Destroys the handles, reporting a failure instead of raising. Idempotent.
void cusolver_shutdown();

// The dtypes the cuSOLVER ops take: the four LAPACK types.
inline bool cusolver_dtype(NanoString dtype) {
    return dtype == ns_float32 || dtype == ns_float64 ||
           dtype == ns_complex64 || dtype == ns_complex128;
}

// The LAPACK letter and the CUDA element type of a dtype, as JIT defines:
// `cusolverDn@LPREFIX@@getrf` names the routine, `Tc` the type it takes.
inline const char* cusolver_letter(NanoString dtype) {
    if (dtype == ns_float32) return "S";
    if (dtype == ns_float64) return "D";
    if (dtype == ns_complex64) return "C";
    return "Z";
}

inline const char* cusolver_ctype(NanoString dtype) {
    if (dtype == ns_float32) return "float";
    if (dtype == ns_float64) return "double";
    if (dtype == ns_complex64) return "cuComplex";
    return "cuDoubleComplex";
}

} // jittor
