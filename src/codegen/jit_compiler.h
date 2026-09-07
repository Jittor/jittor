// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "codegen/op_compiler.h"

namespace jittor {

// Startup values already converted by the backend's flag service. This API
// configures command construction; it never receives or rewrites source code.
// @pyjt(configure_accelerator_compiler)
void configure_accelerator_compiler(const string& path, const string& flags,
    const string& language, const string& source_suffix,
    const vector<string>& remove_flags, bool device_link);

namespace jit_compiler {

jit_op_entry_t compile(
    const string& jit_key, 
    const string& src, 
    const bool is_cuda_op = false,
    const string& extra_flags="");

} // jit_compiler
} // jittor
