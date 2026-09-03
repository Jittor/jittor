// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "utils/str_utils.h"

namespace jittor {

JIT_TEST(token_replace_all_normal_termination) {
    ASSERTop(token_replace_all("foo + foo", "foo", "bar"), ==,
             string("bar + bar"));
    ASSERTop(token_replace_all("untouched", "foo", "bar"), ==,
             string("untouched"));
}

JIT_TEST(token_replace_all_propagates_invalid_patterns) {
    expect_error([&]() { token_replace_all("foo", "", "bar"); });
}

} // jittor
