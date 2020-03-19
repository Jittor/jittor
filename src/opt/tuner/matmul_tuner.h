// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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