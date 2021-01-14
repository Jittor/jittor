// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "cublas_test_op.h"

int cublas_test_entry(int);

namespace jittor {

#ifndef JIT
CublasTestOp::CublasTestOp(int size_mult) : size_mult(size_mult) {
    output = create_output(1, ns_float32);
}

void CublasTestOp::jit_prepare(JK& jk) {
    jk << _CS("[T:float32]");
}

#else // JIT
#ifdef JIT_cpu
void CublasTestOp::jit_run() {
    ASSERT(cublas_test_entry(size_mult)==0);
    output->ptr<T>()[0] = 123;
}
#endif
#endif // JIT

} // jittor
