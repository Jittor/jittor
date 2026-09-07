// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"

namespace jittor {

JIT_TEST(error_categories_are_catchable_and_distinct) {
    bool caught_user = false;
    try {
        USER_CHECK(false) << "invalid public argument";
    } catch (const UserError&) {
        caught_user = true;
    }
    CHECK(caught_user);

    bool caught_invariant = false;
    try {
        INTERNAL_ASSERT(false) << "broken internal state";
    } catch (const InternalInvariantError&) {
        caught_invariant = true;
    }
    CHECK(caught_invariant);

    static_assert(!std::is_base_of<UserError, InternalInvariantError>::value,
        "internal invariant errors must not be caught as user errors");
    static_assert(!std::is_base_of<InternalInvariantError, UserError>::value,
        "user errors must not be caught as internal invariant errors");
}

} // namespace jittor
