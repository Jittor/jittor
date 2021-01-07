// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

struct Tuner {
    string name;
    int confidence;
    loop_option_candidates_t candidates;

    Tuner(const string& name);
    void add_candidate(const string& key, int value);
    virtual ~Tuner();
    virtual void run(PassManager* pm, TunerManager* tm) = 0;
};

}