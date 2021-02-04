// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct EmptyOp : Op {
    EmptyOp(NanoVector shape, NanoString dtype=ns_float32);
    
    const char* name() const override { return "empty"; }
};

} // jittor