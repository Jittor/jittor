// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include "common.h"

namespace jittor {

struct cstr {
    unique_ptr<char[]> ptr;

    inline const char* c_str() const { return ptr ? ptr.get() : ""; }
    inline cstr& operator=(const char* s) {
        auto len = std::strlen(s);
        ptr.reset(new char[len+1]);
        std::memcpy(ptr.get(), s, len+1);
        return *this;
    }
    inline cstr& operator=(const string& s) {
        auto len = s.size();
        ptr.reset(new char[len+1]);
        std::memcpy(ptr.get(), s.c_str(), len+1);
        return *this;
    }
    inline cstr& operator=(cstr&& s) {
        ptr = move(s.ptr);
        return *this;
    }
    inline cstr& operator=(const cstr& s) {
        *this = s.c_str();
        return *this;
    }
    inline cstr(const cstr& s) {
        *this = s.c_str();
    }
    inline cstr() {}
    inline size_t size() const { return ptr ? std::strlen(ptr.get()) : 0; }
};

inline std::ostream& operator<<(std::ostream& os, const cstr& p) {
    return os << p.c_str();
}


inline std::istream& operator>>(std::istream& is, cstr& p) {
    string s;
    is >> s;
    p = s;
    return is;
}

} // jittor
