// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "jit_key.h"

namespace jittor {

JIT_TEST(jit_key) {
    JK& jk = get_jk();
    jk.clear();
    for (int i=0; i<JK::buffer_size/2; i++)
        jk.buffer[i] = i%256;
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

// Writing past the end of the jit key buffer must reach the mprotect guard
// page, and that fault must stop the process: the key selects which compiled
// kernel runs, so an overrun that merely wrapped or truncated would pick the
// wrong kernel and give a wrong answer with no error.
//
// This case is EXPECTED TO KILL THE PROCESS. It used to live inside
// `expect_error()` in the case above, which worked only because the SIGSEGV
// handler threw a C++ exception -- throwing out of a signal handler is
// undefined behaviour that happened to unwind on this ABI, and the case had
// been "passing" on it for years. The handler now reports through write(2) and
// `_exit`s (2.20), so there is nothing to catch and nothing should try:
// tests/compiler/test_jit_tests.py runs this one in a child process and asserts
// on its exit status and on the message the handler prints.
JIT_TEST(jit_key_guard_page) {
    JK& jk = get_jk();
    jk.clear();
    for (int i=0; i<JK::buffer_size; i++)
        jk.buffer[i] = i%256;
    LOGf << "writing past the jit key buffer did not fault: the guard page is"
        << "missing, so an over-long key would be truncated in silence";
}

} // jittor
