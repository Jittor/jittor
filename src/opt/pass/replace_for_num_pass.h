// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

// replace_for_num pass
// T num=opi_x->num;
// for (T i=0; i<num; i++) ...
// ->
// T opi_xshapej = opi_x->shape[j]; ...
// T opi_xstride{DIM-1} = 1;
// T opi_xstride{j} = opi_xstride{j+1} * opi_xshape{j+1}
// for (T i{d}=0; i{d}<opi_shape{d}; i{d}++)
//     i = i{d} * opi_xstride{d}
struct ReplaceForNumPass : Pass {
    ReplaceForNumPass() : Pass("replace_for_num") {};
    void run() override;
};

} // jittor
