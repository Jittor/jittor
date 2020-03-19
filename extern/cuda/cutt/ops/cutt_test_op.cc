// ***************************************************************
// Copyright (c) 2019 Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cutt_test_op.h"
#include "misc/str_utils.h"

#ifdef JIT
#include "cutt.h"
#endif

namespace jittor {

#ifndef JIT
CuttTestOp::CuttTestOp(string cmd) : cmd(cmd) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(1, ns_float32);
}

void CuttTestOp::jit_prepare() {
    add_jit_define("T", ns_float32);
}

#else // JIT
#ifdef JIT_cuda

void CuttTestOp::jit_run() {
    auto args = split(cmd, " ");
    if (!cmd.size()) args.clear();
    vector<char*> v(args.size());
    for (uint i=0; i<args.size(); i++)
        v[i] = &args[i][0];
    output->ptr<T>()[0] = 123;

}
#endif
#endif // JIT

} // jittor
