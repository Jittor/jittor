// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "cub_test_op.h"
#include "misc/str_utils.h"

#ifdef JIT
#include "cub_test.h"
#endif

namespace jittor {

#ifndef JIT
CubTestOp::CubTestOp(string cmd) : cmd(cmd) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CubTestOp::jit_prepare(JK& jk) {
    jk << _CS("[T:float32]");
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
    output->ptr<T>()[0] = 123;
}
#endif
#endif // JIT

} // jittor
