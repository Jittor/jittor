// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cusparse_spmmcoo_op.h"
#include "cusparse_wrapper.h"
using namespace std;

namespace jittor {
#ifndef JIT

CusparseSpmmcooOp::CusparseSpmmcooOp(Var* outputVar_, Var* x_, Var* row_indices_,Var* col_indices_,Var* value_,int A_row_,int A_col_)
    : outputVar(outputVar_), x(x_),row_indices(row_indices_), col_indices(col_indices_), value(value_),A_row(A_row_),A_col(A_col_) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0); 
    flags.set(NodeFlags::_manual_set_vnbb);
    ASSERT(x->dtype().is_float() && outputVar->dtype().is_float()) << "type of two inputs should be the same";
    output = create_output(nullptr, x->dtype());
}

void CusparseSpmmcooOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", x->dtype());
    add_jit_define(jk, "Tindex", col_indices->dtype());
}

#else // JIT

void CusparseSpmmcooOp::jit_run() {
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t &handle_ = cusparse_handle;
    // void*                dBuffer    = NULL;
    // size_t               bufferSize = 0;
    const auto& xs = x->shape;
    const auto& vs = value->shape; 
    const auto& os = outputVar->shape;
    ASSERT(xs==os)<<"matrix A and matrix C size not match";
    ASSERT(A_col==xs[0])<<"matrix A and matrix B size not match";
    auto dtype_A = get_dtype(value->dtype());
    auto dtype_B = get_dtype(x->dtype());
    auto dtype_C = get_dtype(outputVar->dtype());
    auto dtype_index = get_index_dtype(col_indices->dtype());
    checkCudaErrors( cusparseCreateCoo(&matA, A_row, A_col, vs[0], row_indices->ptr<Tindex>(), col_indices->ptr<Tindex>(), value->ptr<T>(), dtype_index, CUSPARSE_INDEX_BASE_ZERO, dtype_A) );
    checkCudaErrors( cusparseCreateDnMat(&matB, xs[0], xs[1], xs[1], x->ptr<T>(), dtype_B, CUSPARSE_ORDER_ROW) );
    checkCudaErrors( cusparseCreateDnMat(&matC, os[0], os[1],os[1], outputVar->ptr<T>(), dtype_C, CUSPARSE_ORDER_ROW) );
    float alpha = 1.0f;
    float beta  = 0.0f;
    // checkCudaErrors( cusparseSpMM_bufferSize(
    //                              handle_,
    //                              CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                              CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
    //                              CUSPARSE_SPMM_ALG_DEFAULT , &bufferSize) );
    // checkCudaErrors( cudaMalloc(&dBuffer, bufferSize) );
    checkCudaErrors( cusparseSpMM(handle_,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, NULL) );
    // checkCudaErrors( cudaFree(dBuffer) );
    checkCudaErrors( cusparseDestroySpMat(matA) );
    checkCudaErrors( cusparseDestroyDnMat(matB) );
    checkCudaErrors( cusparseDestroyDnMat(matC) );
}
#endif // JIT

} // jittor