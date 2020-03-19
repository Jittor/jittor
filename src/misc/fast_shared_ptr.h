// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

template<class T>
struct fast_shared_ptr {
    typedef T value_type;
    pair<uint64, T>* ptr;
    fast_shared_ptr() { ptr = nullptr; }
    fast_shared_ptr(std::nullptr_t) { ptr = nullptr; }

    fast_shared_ptr(T&& a) {
        if (a.size()) {
            ptr = new pair<uint64, T>(1, move(a));
        } else {
            ptr = nullptr;
        }
    }

    fast_shared_ptr(const fast_shared_ptr<T>& other) {
        ptr = other.ptr;
        if (ptr) ptr->first++;
    }

    ~fast_shared_ptr() {
        if (ptr) {
            ptr->first--;
            if (!ptr->first)
                delete ptr;
        }
    }

    void clear() { 
        this->~fast_shared_ptr();
        ptr = nullptr;
     }

    fast_shared_ptr<T>& operator=(std::nullptr_t) {
        clear();
        return *this;
    }

    fast_shared_ptr<T>& operator=(T&& a) {
        this->~fast_shared_ptr();
        new(this) fast_shared_ptr<T>(move(a));
        return *this;
    }

    fast_shared_ptr<T>& operator=(const fast_shared_ptr<T>& other) {
        this->~fast_shared_ptr();
        new(this) fast_shared_ptr<T>(other);
        return *this;
    }

    operator bool() const { return ptr; }
    operator T() const { return ptr ? ptr->second : T(); }
    T& data() const { return ptr->second; }
    uint64 ref_cnt() const { return ptr ? ptr->first : 0; }
};

template<class T>
std::ostream& operator<<(std::ostream& os, const fast_shared_ptr<T>& p) {
    if (p)
        return os << p.ptr->second;
    return os << "null";
}


template<class T>
std::istream& operator>>(std::istream& is, fast_shared_ptr<T>& p) {
    T a;
    is >> a;
    p = move(a);
    return is;
}

} // jittor
