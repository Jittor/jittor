// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include "codegen/jit_key.h"
#include "utils/str_utils.h"

namespace jittor {

// C++14 needs an out-of-line definition for a `static constexpr` member that is
// odr-used, and this one is: it is streamed into the log below and compared
// against in `src/tests/test_jit_key.cc`, both of which bind it to a reference.
// At `-O3` the compiler folds every use and the missing symbol never shows, so
// the omission was invisible until someone built with `JT_BUILD_DEBUG=1` --
// where it becomes `undefined symbol: jittor::JitKey::init_capacity` at import
// and the debug build cannot start at all. Debug symbols are exactly what one
// wants when chasing a segfault, so this made the hardest failures the hardest
// to investigate.
constexpr size_t JitKey::init_capacity;


DEFINE_FLAG(int, jit_key_max_size, 2*1024*1024,
    "Largest jit key, in bytes. A key that would grow past this raises a "
    "catchable error instead of being truncated -- the key selects which "
    "compiled kernel runs, so a truncated one would silently run another "
    "kernel. Before 3.02 the limit was the size of a fixed buffer with an "
    "mprotect'ed guard page at the end, and exceeding it killed the process.");

JitKey::JitKey() {
    // Eager, so `to_cstring()`/`to_string()` are valid before anything is
    // written. One allocation per thread that ever builds a key.
    buffer = (char*)std::malloc(init_capacity);
    CHECK(buffer) << "out of memory allocating the jit key buffer"
        << init_capacity << "bytes";
    capacity = init_capacity;
    check_at = effective_check_at();
    buffer[0] = 0;
}

JitKey::~JitKey() {
    std::free(buffer);
    buffer = nullptr;
    capacity = check_at = 0;
}

void JitKey::grow(size_t n) {
    JT_GBP_SCOPE(gbp_jit_key_grow);
    size_t limit = size_limit();
    size_t need = (size_t)size + n;
    // The key selects which compiled kernel runs, so a key that does not fit
    // must not be truncated or wrapped: either would look up somebody else's
    // kernel and return a wrong answer with nothing reported. It used to run
    // into an mprotect'ed guard page and kill the process from the signal
    // handler; now it is a normal exception the caller can catch, and this JK
    // is left usable -- `size` is unchanged, so `clear()` recovers it.
    USER_CHECK(need <= limit)
        << "jit key too long:" << need << "bytes, the limit is" << limit
        << "(jit_key_max_size)."
        << "\nA key this long means one fused operator grew far past the sizes"
        << "fusion is meant for; jt.flags.no_fuse=1 or a smaller expression"
        << "will avoid it, and jt.flags.jit_key_max_size raises the limit.";
    size_t want = need + tail_slack;
    if (want > capacity) {
        size_t new_capacity = capacity ? capacity : init_capacity;
        while (new_capacity < want) new_capacity *= 2;
        if (new_capacity > limit + tail_slack)
            new_capacity = limit + tail_slack;
        char* grown = (char*)std::realloc(buffer, new_capacity);
        CHECK(grown) << "out of memory growing the jit key buffer to"
            << new_capacity << "bytes";
        buffer = grown;
        capacity = new_capacity;
    }
    // Also the path that picks up a change to the flag.
    check_at = effective_check_at();
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
    ss.imbue(std::locale::classic());
    ss << std::setprecision(std::numeric_limits<float64>::max_digits10)
       << itof(x);
    s = ss.str();
    if (s=="inf") s = "(1.0/0)";
    if (s=="-inf") s = "(-1.0/0)";
    if (s=="nan" || s=="-nan") s = "(0.0/0)";
    // Keep integral values and signed zero as floating-point literals.  JIT
    // kernels are compiled as C++14, where hexadecimal floating literals are
    // not portable, so use a max_digits10 decimal representation instead.
    if (s.find_first_of(".eE/") == string::npos)
        s += ".0";
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
