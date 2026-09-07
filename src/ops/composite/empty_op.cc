// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/composite/array_op.h"
#include "ops/op_register.h"
#include "ops/composite/empty_op.h"

namespace jittor {

EmptyOp::EmptyOp(NanoVector shape, NanoString dtype) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    create_output(shape, dtype);
}

} // jittor