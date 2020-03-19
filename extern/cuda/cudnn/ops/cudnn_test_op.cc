// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "cudnn_test_op.h"
#include "misc/str_utils.h"

int cudnn_test_entry( int argc, char** argv );

namespace jittor {

#ifndef JIT
CudnnTestOp::CudnnTestOp(string cmd) : cmd(move(cmd)) {
    output = create_output(1, ns_float32);
}

void CudnnTestOp::jit_prepare() {
    add_jit_define("T", ns_float32);
}

#else // JIT
#ifdef JIT_cpu
void CudnnTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    ASSERT(cudnn_test_entry(v.size(), &v[0])==0);
    output->ptr<T>()[0] = 123;
}
#endif
#endif // JIT

} // jittor
