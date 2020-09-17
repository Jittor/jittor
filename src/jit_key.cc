// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sys/mman.h>
#include <sstream>
#include "jit_key.h"
#include "misc/str_utils.h"

namespace jittor {

const int page_size = 4*1024;

extern thread_local size_t protected_page;

static size_t get_buffer_end_page(size_t buffer_end) {
    // get the last complete page in buffer
    // 4k align :
    //  |       |       |       |       |
    //  buffer:    xxxxxxxxxxxxxxxxxxxxxxxx
    //                          ^  buffer_end_page
    size_t buffer_end_page = buffer_end - buffer_end % page_size;
    if (buffer_end_page + page_size-1 > buffer_end)
        buffer_end_page -= page_size;
    return buffer_end_page;
}

JitKey::JitKey() {
    auto buffer_end_page = get_buffer_end_page((size_t)&buffer[buffer_size-1]);
    LOGvv << "protect page" << (void*)buffer_end_page;
    ASSERT(0==mprotect((void*)buffer_end_page, page_size, PROT_NONE));
    protected_page = buffer_end_page;
}

JitKey::~JitKey() {
    auto buffer_end_page = get_buffer_end_page((size_t)&buffer[buffer_size-1]);
    LOGvv << "un-protect page" << (void*)buffer_end_page;
    ASSERT(0==
    mprotect((void*)buffer_end_page, page_size, PROT_READ|PROT_WRITE|PROT_EXEC));
    protected_page = 0;
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
    int presum = 0;
    char state=0;
    string key, val;
    for (char c : s) {
        if (c==JK::key) {
            presum++;
            if (presum==1) {
                state = c;
                continue;
            }
        } else
        if (c==JK::val || c==JK::hex_val) {
            if (presum==1 && state==JK::key) {
                state = c;
                continue;
            }
        } else
        if (c==JK::end) {
            presum--;
            if (presum==0) {
                if (state == JK::hex_val)
                    hex_to_dec(val);
                if (startswith(val, "itof"))
                    convert_itof(val);
                jit_keys.emplace_back(move(key), move(val));
                continue;
            }
        }
        if (presum) {
            if (state==JK::key)
                key += c;
            if (state==JK::val || state==JK::hex_val)
                val += c;
        }
    }
    ASSERT(presum==0);
    return jit_keys;
}

thread_local JitKey jk;

} // jittor