// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "op_compiler.h"

namespace jittor {
namespace jit_compiler {

jit_op_entry_t compile(
    const string& jit_key, 
    const string& src, 
    const bool is_cuda_op = false,
    const string& extra_flags="");

} // jit_compiler
} // jittor