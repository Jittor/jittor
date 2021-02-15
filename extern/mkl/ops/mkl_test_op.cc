// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "mkl_test_op.h"

int mkl_test_entry();

namespace jittor {

#ifndef JIT
MklTestOp::MklTestOp() {
    output = create_output(1, ns_float32);
}

void MklTestOp::jit_prepare(JK& jk) {
    jk << _CS("[T:float32]");
}

#else // JIT
#ifdef JIT_cpu
void MklTestOp::jit_run() {
    ASSERT(mkl_test_entry()==0);
    output->ptr<T>()[0] = 123;
}
#endif
#endif // JIT

} // jittor
