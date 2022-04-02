// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#ifdef JIT
#include <cutlass/gemm/device/gemm.h>
#endif
#include "var.h"
#include "cutlass_matmul_op.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

#ifndef JIT

CutlassMatmulOp::CutlassMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == b->dtype().dsize()) << "type of two inputs should be the same";
    c = create_output(nullptr, a->dtype());
}

void CutlassMatmulOp::infer_shape() {
    ASSERTop(a->shape.size(),==,2);
    ASSERTop(b->shape.size(),==,2);
    int n = a->shape[0], m = a->shape[1];
    int m_ = b->shape[0], k = b->shape[1];
    if (trans_a) {
        swap(n, m);
    }
    if (trans_b) {
        swap(m_, k);
    }
    ASSERTop(m,==,m_);
    c->set_shape({n, k});
}

void CutlassMatmulOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << a->dtype();
    jk << _CS("][Trans_a:") << (trans_a ? 'T' : 'N');
    jk << _CS("][Trans_b:") << (trans_b ? 'T' : 'N');
    jk << _CS("][op:") << (a->dtype().dsize() == 4 ? 'S' : 'D');
    jk << ']';
}

#else // JIT
#pragma clang diagnostic ignored "-Wtautological-compare"


static inline cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

void CutlassMatmulOp::jit_run() {
    const T alpha = 1.0f;
    const T beta  = 0.0f;
    LOGi << "herher";
    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto n = as[0];
    auto m = as[1];
    auto k = bs[1];
    if ('@Trans_a'=='T') {
        n = as[1];
        m = as[0];
    }
    if ('@Trans_b'=='T') {
        k = bs[0];
    }
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    // a: [n,m], b: [m,k], c: [n,k]
    cudaError_t result = CutlassSgemmNN(n, k, m, alpha, a->ptr<T>(), '@Trans_a' == 'N' ? m : n, b->ptr<T>(), '@Trans_b' == 'N' ? k : m, beta, c->ptr<T>(), k);    
    assert(result == cudaSuccess);
}
#endif // JIT

} // jittor
