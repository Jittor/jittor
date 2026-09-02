// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "cub_test_op.h"
#include "utils/str_utils.h"

#ifdef JIT
#include "cub_test.h"
#endif

namespace jittor {

#ifndef JIT
CubTestOp::CubTestOp(string cmd) : cmd(cmd) {
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CubTestOp::jit_prepare(JK& jk) {
    jk << "«T:float32";
}

#else // JIT
#ifdef JIT_cuda
void CubTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    ASSERT(cub_test_entry(v.size(), &v[0])==0);
    T result = 123;
    auto status = cudaMemcpy(
        output->ptr<T>(), &result, sizeof(result), cudaMemcpyHostToDevice);
    ASSERT(status == cudaSuccess) << cudaGetErrorString(status);
}
#endif
#endif // JIT

} // jittor
