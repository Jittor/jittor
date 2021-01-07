// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include "var.h"
#include "op.h"
#include "var_holder.h"

namespace jittor {

// @pyjt(number_of_hold_vars)
inline static uint64 get_number_of_hold_vars() {
    return VarHolder::hold_vars.size();
}

// @pyjt(number_of_lived_vars)
inline static int64 get_number_of_lived_vars() {
    return Var::number_of_lived_vars;
}

// @pyjt(number_of_lived_ops)
inline static int64 get_number_of_lived_ops() {
    return Op::number_of_lived_ops;
}

// @pyjt(print_trace)
inline static void __print_trace() {
    print_trace();
}

// @pyjt(grad)
vector<VarHolder*> _grad(VarHolder* loss, const vector<VarHolder*>& targets);

} // jittor
