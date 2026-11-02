// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "cublas_test_op.h"

int cublas_test_entry(int);

namespace jittor {

#ifndef JIT
CublasTestOp::CublasTestOp(int size_mult) : size_mult(size_mult) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CublasTestOp::jit_prepare(JK& jk) {
    jk << "«T:float32";
}

#else // JIT
#ifdef JIT_cuda
void CublasTestOp::jit_run() {
    ASSERT(cublas_test_entry(size_mult)==0);
    T result = 123;
    auto status = cudaMemcpy(output->ptr<T>(), &result, sizeof(result), cudaMemcpyHostToDevice);
    ASSERT(status == cudaSuccess) << cudaGetErrorString(status);
}
#endif
#endif // JIT

} // jittor
