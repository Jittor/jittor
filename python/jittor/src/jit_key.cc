// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <sstream>
#include "jit_key.h"
#include "utils/str_utils.h"

namespace jittor {

#ifndef _WIN32
EXTERN_LIB thread_local size_t protected_page;

static size_t get_buffer_end_page(size_t buffer_end) {
    // get the last complete page in buffer
    // 4k align :
    //  |       |       |       |       |
    //  buffer:    xxxxxxxxxxxxxxxxxxxxxxxx
    //                          ^  buffer_end_page
    size_t buffer_end_page = buffer_end - buffer_end % getpagesize();
    if (buffer_end_page + getpagesize()-1 > buffer_end)
        buffer_end_page -= getpagesize();
    return buffer_end_page;
}
#endif

JitKey::JitKey() {
#ifndef _WIN32
    auto buffer_end_page = get_buffer_end_page((size_t)&buffer[buffer_size-1]);
    LOGvv << "protect page" << (void*)buffer_end_page;
    ASSERT(0==mprotect((void*)buffer_end_page, getpagesize(), PROT_NONE));
    protected_page = buffer_end_page;
#endif
}

JitKey::~JitKey() {
#ifndef _WIN32
    auto buffer_end_page = get_buffer_end_page((size_t)&buffer[buffer_size-1]);
    LOGvv << "un-protect page" << (void*)buffer_end_page;
    mprotect((void*)buffer_end_page, getpagesize(), PROT_READ|PROT_WRITE|PROT_EXEC);
    protected_page = 0;
#endif
}

static void hex_to_dec(string& s) {
    // check s is hex or not, if yes, convert to dec
    if (!s.size()) return;
    unsigned int x;
    std::stringstream ss;
    ss << std::hex << s;
    ss >> x;
    s = S(x);
}

static void convert_itof(string& s) {
    uint64 x;
    std::stringstream ss;
    // itof(0x...)
    //        ^ ^
    //        7
    ASSERT(s.size()>=8);
    ss << std::hex << s.substr(7, s.size()-7-1);
    ASSERT(ss >> x);
    ss.str(""); ss.clear();
    ss << std::hexfloat << itof(x);
    s = ss.str();
    // 0x0p+0 ---> 0x0p0
    if (s.find("p+") != string::npos)
        s.erase(s.find("p+")+1, 1); 
    if (s=="inf") s = "(1.0/0)";
    if (s=="-inf") s = "(-1.0/0)";
    if (s=="nan" || s=="-nan") s = "(0.0/0)";
}

vector<pair<string,string>> parse_jit_keys(const string& s) {
    vector<pair<string,string>> jit_keys;
    auto sp = split(s, JitKey::key);
    for (auto& ss : sp) {
        if (!ss.size()) continue;
        string key, val;
        char state=0;
        for (auto c : ss) {
            if (state == 0 && 
                (c==JK::val || c==JK::hex_val)) {
                state = c;
                continue;
            }
            if (state == 0) key += c;
            else val += c;
        }
        if (state == JK::hex_val)
            hex_to_dec(val);
        if (startswith(val, "itof"))
            convert_itof(val);
        jit_keys.emplace_back(move(key), move(val));
    }
    return jit_keys;
}

thread_local JitKey jk;

JK& get_jk() {
    return jk;
}

} // jittor