// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "node.h"

namespace jittor {

JIT_TEST(node_liveness_contract) {
    NodeLiveness state;
    CHECK(state.need_free());

    CHECK(state.forward.own());
    CHECK(state.backward.own());
    CHECK(state.pending.own());
    CHECK(!state.need_free());
    CHECK(!state.forward.own());
    ASSERTop(state.forward.count(),==,2);

    CHECK(!state.forward.release());
    CHECK(state.pending.release());
    CHECK(!state.need_free());
    CHECK(state.backward.release());
    CHECK(state.need_free());
    CHECK(state.forward.release());

    // These checks use Jittor's always-on CHECK/ASSERT path, not C assert or
    // NODE_MEMCHECK, so release builds enforce them too.
    expect_error([&]() { state.forward.release(); });
    state.assert_expected(0, 0, 0, &state);
    expect_error([&]() { state.assert_expected(1, 0, 0, &state); });
}

} // namespace jittor
