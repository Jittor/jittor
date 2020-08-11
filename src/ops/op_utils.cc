// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/op_register.h"
#include "var.h"

namespace jittor {

static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_unary = get_op_info("unary")
    .get_constructor<VarPtr, Var*, NanoString>();
static auto make_broadcast_to = get_op_info("broadcast_to")
    .get_constructor<VarPtr, Var*, Var*, NanoVector>();

VarPtr make_number(float number, Var* x) {
    VarPtr nums = make_array(&number, 1, ns_float32);
    nums = make_broadcast_to(nums, x, {});
    return make_unary(nums, x->dtype());
}

static void init() {
    op_registe({"number", "", "", {{&typeid(&make_number), (void*)&make_number}}});
}
__attribute__((unused)) static int caller = (init(), 0);

} // jittor
