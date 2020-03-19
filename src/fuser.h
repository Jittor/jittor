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

namespace jittor {

void count_fuse(int64_t tt, int start_var_num, const vector<Op*>& ops, const vector<Var*>& vars, vector<int> &father, vector<int> &var_fused);

} // jittor
