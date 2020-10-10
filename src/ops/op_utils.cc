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
    union Number {
        float32 f32;
        float64 f64;
        int32 i32;
        int64 i64;
    } v;
    if (x->dtype() == ns_float32) v.f32 = number; else
    if (x->dtype() == ns_float64) v.f64 = number; else
    if (x->dtype() == ns_int32) v.i32 = number; else
    if (x->dtype() == ns_int64) v.i64 = number; else {
        VarPtr nums = make_array(&number, 1, ns_float32);
        nums = make_broadcast_to(nums, x, {});
        return make_unary(nums, x->dtype());
    }
    VarPtr nums = make_array(&v, 1, x->dtype());
    return make_broadcast_to(nums, x, {});
}

static void init() {
    op_registe({"number", "", "", {{&typeid(&make_number), (void*)&make_number}}});
}
__attribute__((unused)) static int caller = (init(), 0);

} // jittor
