// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "cublas_acc_matmul_op.h"
#include "cublas_wrapper.h"

using namespace std;

namespace jittor {

extern int use_tensorcore;

#ifndef JIT

CublasAccMatmulOp::CublasAccMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b, int stride_a, int stride_b, int offset_a, int offset_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b),stride_a(stride_a),stride_b(stride_b),offset_a(offset_a),offset_b(offset_b) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_manual_set_vnbb);
    a->flags.set(NodeFlags::_needed_by_backward);
    b->flags.set(NodeFlags::_needed_by_backward);
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == b->dtype().dsize()) << "type of two inputs should be the same";
    c = create_output(nullptr, a->dtype());
}

void CublasAccMatmulOp::infer_shape() {
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
    if(stride_a != -1)
        n = stride_a;
    if(stride_b != -1)
        k = stride_b;
    c->set_shape({n, k});
}

void CublasAccMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << a->dtype();
    jk << "«Trans_a:" << (trans_a ? 'T' : 'N');
    jk << "«Trans_b:" << (trans_b ? 'T' : 'N');
    jk << "«op:" << (a->dtype().dsize() == 2? 'H' : (a->dtype().dsize() == 4 ? 'S' : 'D'));
}

#else // JIT
#pragma clang diagnostic ignored "-Wtautological-compare"

void CublasAccMatmulOp::jit_run() {
    cublasHandle_t& handle_ = cublas_handle;
    const T alpha = 1.0f;
    const T beta  = 0.0f;

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

    // a: [n,m], b: [m,k], c: [n,k]
    #if CUDART_VERSION >= 11000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    if (use_tensorcore>=3) {
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (use_tensorcore==2) {
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
    } else if (use_tensorcore==1) {
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
    if (a->dtype() == ns_float16
        || b->dtype() == ns_float16 || c->dtype() == ns_float16) {
        computeType = CUBLAS_COMPUTE_16F;
    }
    #else
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    cudaDataType_t computeType = get_dtype(c->dtype());
    if (use_tensorcore) {
        algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    if (a->dtype() == ns_float16
        || b->dtype() == ns_float16 || c->dtype() == ns_float16) {
        computeType = CUDA_R_16F;
        algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    #endif
    int ldb, lda;
    ldb = '@Trans_b' == 'N' ? k : m;
    lda = '@Trans_a' == 'N' ? m : n;
    if(stride_b != -1)
        k = stride_b;
    // if(stride_a != -1)
    //     n = stride_a;
    checkCudaErrors(cublasGemmEx(handle_, 
    CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a, 
    k, n, m, &alpha, 
    b->ptr<T>() + offset_b,get_dtype(b->dtype()), ldb, 
    a->ptr<T>() + offset_a,get_dtype(a->dtype()), lda, &beta, 
    c->ptr<T>(),get_dtype(c->dtype()), k,
    computeType, algo));
    // checkCudaErrors(cublas@op@@gemm(handle_, 
    // CUBLAS_OP_@Trans_b, CUBLAS_OP_@Trans_a, 
    // k, n, m, &alpha, 
    // b->ptr<T>(), '@Trans_b' == 'N' ? k : m, 
    // a->ptr<T>(), '@Trans_a' == 'N' ? m : n, &beta, 
    // c->ptr<T>(), k));

    
}
#endif // JIT

} // jittor
