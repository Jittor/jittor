// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include <acl/acl.h>

std::string acl_error_to_string(aclError error);

namespace jittor {

EXTERN_LIB uint64_t acl_jittor_tid;

void acl_jittor_op_compiler(string& filename, string& src, bool is_acl, string& extra_flags);

}
