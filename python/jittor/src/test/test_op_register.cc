// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "op.h"
#include "var.h"
#include "ops/op_register.h"

namespace jittor {

// The registry is read by the name truncated at the first '.' -- that is how
// `name_ex()` spellings like "binary.add" resolve to the "binary" op. It used
// to be *written* by the untruncated name, so the two agreed only as long as
// no registered name contained a dot. One that did went in under a key nothing
// could ever look up, and every spelling of it reported "Op not found".
JIT_TEST(op_register_reads_and_writes_the_same_key) {
    const char* dotted = "jit_test_op.variant";
    op_registe({dotted, "", ""});

    ASSERT(has_op(dotted)) << "registered under one key, looked up under another";
    ASSERT(has_op("jit_test_op")) << "the truncated spelling must resolve too";
    ASSERTop(get_op_info(dotted).name,==,string(dotted));
    ASSERTop(get_op_info("jit_test_op").name,==,string(dotted));

    // and the ordinary case is unchanged: "binary.add" finds "binary"
    ASSERT(has_op("binary.add"));
    ASSERTop(get_op_info("binary.add").name,==,string("binary"));
}

// A constructor resolved on first call, not at load time. The point is what
// does NOT happen at construction: no registry lookup, so no dependency on
// this translation unit's static initialiser running after the registry's.
JIT_TEST(op_constructor_resolves_lazily) {
    auto missing = op_constructor<VarPtr, Var*>("jit_test_no_such_op");
    // constructing it asked the registry nothing
    ASSERT(!(bool)missing);
    // and only calling it fails, where there is someone to catch it
    expect_error([&]() { missing(nullptr); });

    auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
    ASSERT((bool)make_unary);
    VarPtr a({4}, "float32");
    auto b = make_unary(a, ns_float64);
    ASSERT(b->dtype() == ns_float64);
}

}
