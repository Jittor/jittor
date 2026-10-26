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

#ifdef JIT
#include <cuda_runtime.h>
#endif

namespace jittor {

#ifndef JIT
CublasTestOp::CublasTestOp(int size_mult) : size_mult(size_mult) {
    // The library this op exercises is a CUDA one, and the op module is built
    // with the accelerator implementation only: compile_extern passes
    // backend="accelerator" for every *_ops library, so no CPU implementation
    // is registered. Without these flags the inherited CPU flag stayed set,
    // the executor looked up an implementation the module never registered,
    // and the op failed with "No kernel registered for cublas_test on cpu".
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
    // The output Var lives in device memory: a host store through
    // output->ptr<T>() would write the host address space, not the tensor.
    // CubTestOp, the one of the three that worked, copies the value instead.
    T result = 123;
    auto status = cudaMemcpy(
        output->ptr<T>(), &result, sizeof(result), cudaMemcpyHostToDevice);
    ASSERT(status == cudaSuccess) << cudaGetErrorString(status);
}
#endif
#endif // JIT

} // jittor
