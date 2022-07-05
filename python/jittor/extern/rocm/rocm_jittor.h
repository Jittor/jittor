// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

void rocm_jittor_op_compiler(string& filename, string& src, bool is_rocm, string& extra_flags);

}
