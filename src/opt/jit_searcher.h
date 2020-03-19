// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

DECLARE_FLAG(int, jit_search_kernel);

struct Searcher {
    OpCompiler* oc;
    int64_t timeout, best_time;
    loop_options_t best_choices;
    
    Searcher(OpCompiler* oc);
    void reset();
    int64_t get_time_of_current_choices();
    void search(const loop_option_candidates_t& candidates);
};

}