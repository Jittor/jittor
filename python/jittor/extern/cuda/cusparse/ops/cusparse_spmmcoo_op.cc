// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers:  Shizhan Lu <578752274@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cusparse_spmmcoo_op.h"
#include "cusparse_wrapper.h"
#include "executor.h"
using namespace std;

namespace jittor {
#ifndef JIT

CusparseSpmmcooOp::CusparseSpmmcooOp(Var* outputVar_, Var* x_, Var* row_indices_,Var* col_indices_,Var* value_,int A_row_,int A_col_,bool trans_A_,bool trans_B_)
    : outputVar(outputVar_), x(x_),row_indices(row_indices_), col_indices(col_indices_), value(value_),A_row(A_row_),A_col(A_col_),trans_A(trans_A_),trans_B(trans_B_) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0); 
    set_flag(OpFlags::_manual_set_vnbb);
    // The check now says what the message always claimed. jit_run reads the
    // values through ptr<T> with T taken from x, so a value array of another
    // float dtype was reinterpreted rather than converted.
    USER_CHECK(x->dtype().is_float()) << "spmm needs a float dtype, got" << x->dtype();
    USER_CHECK(x->dtype() == outputVar->dtype() && x->dtype() == value->dtype())
        << "spmm needs x, values and output of the same dtype, got"
        << x->dtype() << value->dtype() << outputVar->dtype();
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
    cusparseHandle_t handle_ = cusparse_bind_stream();
    const auto& xs = x->shape;
    const auto& vs = value->shape; 
    const auto& os = outputVar->shape;
    USER_CHECKop(xs,==,os)
        << "cuSPARSE COO matrix A and matrix C sizes must match, got " << xs << " and " << os;
    USER_CHECKop(A_col,==,xs[0])
        << "cuSPARSE COO matrix A columns must match matrix B rows, got " << A_col
        << " and " << xs[0];
    auto dtype_A = get_dtype(value->dtype());
    auto dtype_B = get_dtype(x->dtype());
    auto dtype_C = get_dtype(outputVar->dtype());
    auto dtype_index = get_index_dtype(col_indices->dtype());
    checkCudaErrors( cusparseCreateCoo(&matA, A_row, A_col, vs[0], row_indices->ptr<Tindex>(), col_indices->ptr<Tindex>(), value->ptr<T>(), dtype_index, CUSPARSE_INDEX_BASE_ZERO, dtype_A) );
    checkCudaErrors( cusparseCreateDnMat(&matB, xs[0], xs[1], xs[1], x->ptr<T>(), dtype_B, CUSPARSE_ORDER_ROW) );
    checkCudaErrors( cusparseCreateDnMat(&matC, os[0], os[1],os[1], outputVar->ptr<T>(), dtype_C, CUSPARSE_ORDER_ROW) );
    // Compute type and scalars follow the dtype together: cusparseSpMM reads
    // alpha/beta as raw memory of the compute type. Both were fixed at
    // CUDA_R_32F and `float`, so an fp64 product was computed in single
    // precision *and* scaled by whatever eight bytes started at &alpha.
    auto compute_type = get_compute_dtype(value->dtype());
    JT_CUSPARSE_COMPUTE_TYPE(T) alpha = 1, beta = 0;

    // The buffer-size query was commented out and NULL passed in its place.
    // CUSPARSE_SPMM_ALG_DEFAULT picks the algorithm, and the ones that want an
    // external buffer then read through a null pointer -- undefined behaviour
    // that happened to survive on the shapes anyone had tried, and unlike the
    // CSR variant next door, which does ask.
    size_t bufferSize = 0;
    checkCudaErrors( cusparseSpMM_bufferSize(
                                 handle_,
                                 get_trans_type(trans_A),
                                 get_trans_type(trans_B),
                                 &alpha, matA, matB, &beta, matC, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    // cudaFree is a synchronizing call, so the old cudaMalloc/cudaFree pair
    // meant a full-device sync on every SpMM. temp_allocator is what the other
    // ops in this tree use.
    void* dBuffer = nullptr;
    size_t dBufferAllocation = 0;
    if (bufferSize > 0)
        dBuffer = exe.temp_allocator->alloc(bufferSize, dBufferAllocation);

    // The choice this op made, readable from a test: which compute type, and
    // where the external buffer came from. Both were invisible before, and
    // both were wrong (fp32 compute for fp64; a NULL buffer for COO).
    LOGvvv << "cusparse_spmmcoo select: compute=" >> cusparse_compute_type_name(compute_type)
        << "buffer_bytes=" >> bufferSize
        << "buffer_from=" >> (dBuffer ? exe.temp_allocator->name() : "none");

    checkCudaErrors( cusparseSpMM(handle_,
                                 get_trans_type(trans_A),
                                 get_trans_type(trans_B),
                                 &alpha, matA, matB, &beta, matC, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

    if (dBuffer)
        exe.temp_allocator->free(dBuffer, bufferSize, dBufferAllocation);
    checkCudaErrors( cusparseDestroySpMat(matA) );
    checkCudaErrors( cusparseDestroyDnMat(matB) );
    checkCudaErrors( cusparseDestroyDnMat(matC) );
}
#endif // JIT

} // jittor
