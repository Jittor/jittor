// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/array_op.h"
#include "ops/op_register.h"
#include "ops/empty_op.h"

namespace jittor {

EmptyOp::EmptyOp(NanoVector shape, NanoString dtype) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    create_output(shape, dtype);
}

} // jittor