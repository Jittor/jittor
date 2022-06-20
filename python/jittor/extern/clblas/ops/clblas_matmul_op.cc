// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "var.h"
#include "clblas_matmul_op.h"
#include <clBLAS.h>
#include "clblas_warper.h"
#include "opencl_warper.h"

using namespace std;

namespace jittor {

#ifndef JIT

ClblasMatmulOp::ClblasMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b)
    : a(a), b(b), trans_a(trans_a), trans_b(trans_b) {
    flags.set(NodeFlags::_cuda);
    // TODO: support int8 * int8
    ASSERT(a->dtype().is_float() && b->dtype().is_float()) << "type of two inputs should be the same";
    // TODO: support diffrent input type
    ASSERT(a->dtype().dsize() == 4 && b->dtype().dsize() == 4) << "support float32 only now.";
    c = create_output(nullptr, a->dtype());
}

void ClblasMatmulOp::infer_shape() {
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

void ClblasMatmulOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << a->dtype();
    jk << _CS("][Trans_a:") << (trans_a ? 'T' : 'N');
    jk << _CS("][Trans_b:") << (trans_b ? 'T' : 'N') << ']';
}

#else // JIT
#pragma clang diagnostic ignored "-Wtautological-compare"
void ClblasMatmulOp::jit_run() {
    const auto& as = a->shape;
    const auto& bs = b->shape;
    auto M = as[0];
    auto K = as[1];
    auto N = bs[1];
    auto lda = as[1];
    auto ldb = bs[1];

    if ('@Trans_a'=='T') {
        M = as[1];
        K = as[0];
    }
    if ('@Trans_b'=='T') {
        N = bs[0];
    }

    // cout << "ClblasMatmulOp::jit_run()" << " " << a->shape << " " << b->shape << " " << c->shape << " " << trans_a << " " << trans_b << " " << M << " " << K << " " << N << endl;

    cl_int err = 0;
    // // cl_mem bufA, bufB, bufC;
    // cl_event event = NULL;
    clblasOrder order = clblasRowMajor;
    clblasTranspose transA = clblasNoTrans;
    if (trans_a) transA = clblasTrans;
    clblasTranspose transB = clblasNoTrans;
    if (trans_b) transB = clblasTrans;

    /* Prepare OpenCL memory objects and place matrices inside them. */
    // bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, a->size,
    //                       NULL, &err);
    // bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, b->size,
    //                       NULL, &err);
    // bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, c->size,
    //                       NULL, &err);

    // err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
    //     a->size, a->mem_ptr, 0, NULL, NULL);
    // err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
    //     b->size, b->mem_ptr, 0, NULL, NULL);
    // err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
    //     c->size, c->mem_ptr, 0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer() failed with %d\n", err);
    }
    // printf("a  %f\n",*(float*)a->mem_ptr);

    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm(order, transA, transB, M, N, K,
                         1, *(cl_mem*)(a->mem_ptr), 0, lda,
                         *(cl_mem*)(b->mem_ptr), 0, ldb, 0,
                         *(cl_mem*)(c->mem_ptr), 0, N,
                         1, &opencl_queue, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        LOGw << clblasInvalidValue;
        printf("clblasSgemmEx() failed with %d\n", err);
    }
    // else {
        /* Wait for calculations to be finished. */
        // printf("clblasSgemmEx() success with %d\n", err);
        // err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        // err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
        //                           c->size,
        //                           c->mem_ptr, 0, NULL, NULL);
    // }

    /* Release OpenCL events. */
    // clReleaseEvent(event);

    /* Release OpenCL memory objects. */
    // clReleaseMemObject(bufC);
    // clReleaseMemObject(bufB);
    // clReleaseMemObject(bufA);
}
#endif // JIT

} // jittor
