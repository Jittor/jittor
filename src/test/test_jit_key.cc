// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "jit_key.h"

namespace jittor {

JIT_TEST(jit_key) {
    jk.clear();
    for (int i=0; i<JK::buffer_size/2; i++)
        jk.buffer[i] = i%256;
    expect_error([]() {
        for (int i=0; i<JK::buffer_size; i++)
            jk.buffer[i] = i%256;
    });
    std::cerr << "get segfault, ok" << std::endl;

    jk << JK::key << "key" << JK::val << "value" << JK::end;
    jk << JK::key << "key" << JK::val << JK::hex(0x123123) << JK::end;
    jk << JK::key << "key" << JK::val << JK::hex1(0x123123) << JK::end;
    jk << JK::key << "key" << JK::val << JK::hex2(0x123123) << JK::end;
    jk << JK::key << "key" << JK::val << JK::Oxhex(0x123123) << JK::end;
    jk << JK::key << "key" << JK::val << JK::Oxhex1(0x123123) << JK::end;
    jk << JK::key << "key" << JK::val << JK::Oxhex2(0x123123) << JK::end;
    string key = "[key:value][key:123123][key:3][key:23][key:0x123123][key:0x3][key:0x23]";
    ASSERTop(jk.to_string(),==,key);
    auto keys = parse_jit_keys("[a:11][b:22][a[3]:b::[x]][x=11][f=itof(0x0)]");
    vector<pair<string,string>> k2 = 
        {{"a","11"},{"b","22"},{"a[3]","b::[x]"},{"x","17"},{"f","0"}};
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
    add_jit_define("f", 0.01);
    add_jit_define("f", 0.5);
    add_jit_define("f", 1.0/0);
    add_jit_define("f", -1.0/0);
    add_jit_define("f", 0.0/0);
    keys = parse_jit_keys(jk.to_string());
    k2 = {{"f","0x1.47ae147ae147bp-7"}, 
        {"f","0x1p-1"},
        {"f","(1.0/0)"},
        {"f","(-1.0/0)"},
        {"f","(0.0/0)"},
        };
    ASSERTop(keys,==,k2);

}

} // jittor