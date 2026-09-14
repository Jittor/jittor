// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstdio>
#include "codegen/jit_key.h"

namespace jittor {

JIT_TEST(jit_key) {
    JK& jk = get_jk();
    // Dirty the buffer first: nothing below may depend on it starting out
    // zeroed. This used to store into `jk.buffer` directly, which was only
    // possible while the buffer was a fixed 2 MB array.
    jk.clear();
    for (int i=0; i<256; i++)
        jk << "0123456789abcdef";
    jk.clear();
    jk << JK::key << "key" << JK::val << "value";
    jk << JK::key << "key" << JK::val << JK::hex(0x123123);
    jk << JK::key << "key" << JK::val << JK::hex1(0x123123);
    jk << JK::key << "key" << JK::val << JK::hex2(0x123123);
    jk << JK::key << "key" << JK::val << JK::Oxhex(0x123123);
    jk << JK::key << "key" << JK::val << JK::Oxhex1(0x123123);
    jk << JK::key << "key" << JK::val << JK::Oxhex2(0x123123);
    string key = "«key:value«key:123123«key:3«key:23«key:0x123123«key:0x3«key:0x23";
    ASSERTop(jk.to_string(),==,key);
    auto keys = parse_jit_keys("«a:11«b:22«a[3]:b::[x]«x=11«f:itof(0x0)");
    vector<pair<string,string>> k2 = 
        {{"a","11"},{"b","22"},{"a[3]","b::[x]"},{"x","17"},{"f","0.0"}};
    ASSERTop(keys,==,k2);
    jk.clear();jk << 0x0;
    ASSERT(jk.to_string()=="0");
    for (int i=1; i<63; i++) {
        jk.clear();
        jk << ((1ll << i)-1);
        ASSERT(jk.size==(i-1)/4+1);
        jk.clear();
        jk << -((1ll << i)-1);
        ASSERT(jk.size==(i-1)/4+2);
    }

    jk.clear();
    add_jit_define(jk, "f", 0.01);
    add_jit_define(jk, "f", 0.5);
    add_jit_define(jk, "f", 0.7);
    add_jit_define(jk, "f", -0.7);
    add_jit_define(jk, "f", itof(0x8000000000000000ull));
    #ifndef _MSC_VER
    add_jit_define(jk, "f", 1.0/0);
    add_jit_define(jk, "f", -1.0/0);
    add_jit_define(jk, "f", 0.0/0);
    #endif
    keys = parse_jit_keys(jk.to_string());
    k2 = {{"f","0.01"},
        {"f","0.5"},
        {"f","0.69999999999999996"},
        {"f","-0.69999999999999996"},
        {"f","-0.0"},
        {"f","(1.0/0)"},
        {"f","(-1.0/0)"},
        {"f","(0.0/0)"},
        };
    ASSERTop(keys,==,k2);

}

// A key far larger than the buffer started out as must still be assembled
// exactly: the buffer grows, and growth must not lose or reorder bytes.
//
// It starts at `init_capacity`, so this crosses several reallocations. The
// content is checked rather than only the length, because a growth that copied
// the wrong number of bytes would still produce a plausible `size`.
JIT_TEST(jit_key_grows) {
    JK& jk = get_jk();
    jk.clear();
    ASSERTop(jk.size,==,0);
    string expect;
    char hexbuf[32];
    for (int i=0; i<40000; i++) {
        jk << JK::key << "opkey" << i << JK::val << "add";
        // `operator<<(JK&, int)` is variable-width lowercase hex without
        // leading zeros, i.e. what %x prints.
        snprintf(hexbuf, sizeof(hexbuf), "%x", i);
        expect += "«opkey";
        expect += hexbuf;
        expect += ":add";
    }
    ASSERTop((size_t)jk.size,>,JK::init_capacity);
    ASSERTop(jk.to_string(),==,expect);
    jk.finilize();
    ASSERTop(string(jk.to_cstring()),==,expect);
}

// Writing past the size limit must be *reported*, not truncated and not fatal.
//
// The key selects which compiled kernel runs, so a key that silently wrapped
// or got cut short would look up an unrelated kernel and return a wrong answer
// with nothing printed. Until 3.02 the check was an mprotect'ed guard page at
// the end of a fixed 2 MB array: an overrun raised SIGSEGV, jittor's handler
// wrote "Accessing protect pages, maybe jit_key too long" and `_exit`ed, and
// tests/codegen/test_jit_tests.py had to run the case in a child process and
// assert on its exit status because there was nothing to catch. There is now:
// the limit is enforced before every store and raises `UserError`, which
// reaches Python as a RuntimeError.
JIT_TEST(jit_key_overflow) {
    JK& jk = get_jk();
    size_t limit = JK::size_limit();
    jk.clear();
    string chunk(64*1024, 'x');
    bool caught = false;
    string message;
    try {
        for (size_t written=0; written <= limit + chunk.size();
                written += chunk.size())
            jk << chunk;
    } catch (const UserError& e) {
        caught = true;
        message = e.what();
    }
    ASSERT(caught) << "an over-long jit key was accepted";
    ASSERT(message.find("jit key too long") != string::npos) << message;
    // Nothing was half-applied: the refused store did not advance `size`, and
    // what did fit is still inside the limit and still a whole number of
    // chunks.
    ASSERTop((size_t)jk.size,<=,limit);
    ASSERTop(jk.size % (int64)chunk.size(),==,0);
    // And the JK is reusable afterwards -- the point of not dying.
    jk.clear();
    jk << JK::key << "key" << JK::val << "value";
    ASSERTop(jk.to_string(),==,string("«key:value"));
}

} // jittor
