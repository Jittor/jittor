// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "cudnn_test_op.h"
#include "utils/str_utils.h"

int cudnn_test_entry( int argc, char** argv );

#ifdef JIT
#include <cuda_runtime.h>
#endif

namespace jittor {

#ifndef JIT
CudnnTestOp::CudnnTestOp(string cmd) : cmd(move(cmd)) {
    // See cublas_test_op.cc: the accelerator-only op module registers no CPU
    // implementation, and the inherited CPU flag sent the op there anyway.
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CudnnTestOp::jit_prepare(JK& jk) {
    jk << "«T:float32";
}

#else // JIT
#ifdef JIT_cuda
void CudnnTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    ASSERT(cudnn_test_entry(v.size(), &v[0])==0);
    // Device-resident output: copy the value in, do not store through a host
    // pointer. See cublas_test_op.cc.
    T result = 123;
    auto status = cudaMemcpy(
        output->ptr<T>(), &result, sizeof(result), cudaMemcpyHostToDevice);
    ASSERT(status == cudaSuccess) << cudaGetErrorString(status);
}
#endif
#endif // JIT

} // jittor
