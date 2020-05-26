// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
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

struct ConvTuner : Tuner {
    ConvTuner() : Tuner("conv") {}
    void forwardTune(FusedOp* fop);
    void run(PassManager* pm, TunerManager* tm);
};

}