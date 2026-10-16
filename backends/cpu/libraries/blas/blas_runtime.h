// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "type/nano_string.h"

namespace jittor {

// One CBLAS GEMM per matrix of a row-major batch: c[i] = op(a[i]) @ op(b[i]),
// a [batch, n, m] (or [batch, m, n] with trans_a), b [batch, m, k] (or
// [batch, k, m] with trans_b), c [batch, n, k], all dense.
//
// Which CBLAS this links -- numpy's bundled OpenBLAS by default, or the one
// JT_BUILD_BLAS_LIB_PATH names -- is decided by `setup_blas` in
// python/jittor/build/compile_extern.py, which also picks the symbol names
// and the integer width (ILP64 or LP64) the library was built with.
EXTERN_LIB void blas_gemm_execute(NanoString dtype, int64 batch, int64 n, int64 m, int64 k,
                                  bool trans_a, bool trans_b,
                                  const void* a, const void* b, void* c);

// The dtypes `blas_gemm_execute` takes.
inline bool blas_gemm_dtype(NanoString dtype) {
    return dtype == ns_complex64 || dtype == ns_complex128;
}

} // jittor
