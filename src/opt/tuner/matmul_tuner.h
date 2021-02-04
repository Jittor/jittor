// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "var.h"
#include "opt/tuner_manager.h"

namespace jittor {

struct MatmulTuner : Tuner {
    MatmulTuner() : Tuner("matmul") {}
    void run(PassManager* pm, TunerManager* tm);
};

}