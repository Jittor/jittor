// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "codegen/opt/tuner/tuner.h"
#include "codegen/opt/jit_searcher.h"

namespace jittor {

struct TunerManager {
    OpCompiler* oc;
    Searcher searcher;
    Tuner* best_tuner;

    vector<unique_ptr<Tuner>> tuners;

    TunerManager(OpCompiler* oc);
    string tune();

    // run and store a tuner, return confidence
    template <class T> void run_tuner(PassManager* pm);
};

}