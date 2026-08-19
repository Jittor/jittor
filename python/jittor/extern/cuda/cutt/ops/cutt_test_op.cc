// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cutt_test_op.h"
#include "utils/str_utils.h"

#ifdef JIT
#include "cutt.h"
#include "helper_cuda.h"
#endif

namespace jittor {

#ifndef JIT
CuttTestOp::CuttTestOp(string cmd) : cmd(cmd) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CuttTestOp::jit_prepare(JK& jk) {
    jk << "«T:float32";
}

#else // JIT
#ifdef JIT_cuda

void CuttTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    // The output lives in device memory: cuda_managed_allocator is off by
    // default, so assigning through the pointer from host code faults.
    T value = 123;
    checkCudaErrors(cudaMemcpy(output->ptr<T>(), &value, sizeof(T),
                               cudaMemcpyHostToDevice));
}
#endif
#endif // JIT

} // jittor
