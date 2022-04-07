// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct SharedReducePass : Pass {
    SharedReducePass() : Pass("shared_reduce") {};
    void run() override;
};

} // jittor
