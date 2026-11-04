// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/op_register.h"
#include "core/var.h"

namespace jittor {

static auto make_array = op_constructor<VarPtr, const void*, NanoVector, NanoString>("array");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_broadcast_to = op_constructor<VarPtr, Var*, Var*, NanoVector>("broadcast_to");

// The constant is made where `x` is, not on the current device: a gradient
// seed `make_number(1, loss)` used to land on the ambient GPU while the loss
// lived on another, and so did every gradient grown from it (`grad(x, x)` for
// an x on cuda:0 under cuda:1 came back on cuda:1). A pending array is moved
// the way Op::propagate_device moves a pending scalar.
static void place_like(Var* constant, Var* x) {
    constant->device_id = x->device_id;
    if (x->placement.explicit_backend) constant->placement = x->placement;
}

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
        VarPtr nums = make_array(&number, {}, ns_float32);
        place_like(nums.ptr, x);
        nums = make_broadcast_to(nums, x, {});
        return make_unary(nums, x->dtype());
    }
    VarPtr nums = make_array(&v, {}, x->dtype());
    place_like(nums.ptr, x);
    return make_broadcast_to(nums, x, {});
}

static void init() {
    op_registe({"number", "", "", {op_constructor_entry(&make_number)}});
}
static int caller = (init(), 0);

} // jittor
