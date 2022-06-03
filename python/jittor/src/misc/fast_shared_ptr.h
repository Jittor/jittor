// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
    inline fast_shared_ptr() { ptr = nullptr; }
    inline fast_shared_ptr(std::nullptr_t) { ptr = nullptr; }

    inline fast_shared_ptr(T&& a) {
        if (a.size()) {
            ptr = new pair<uint64, T>(1, move(a));
        } else {
            ptr = nullptr;
        }
    }

    inline fast_shared_ptr(const fast_shared_ptr<T>& other) {
        ptr = other.ptr;
        if (ptr) ptr->first++;
    }

    inline ~fast_shared_ptr() {
        if (ptr) {
            ptr->first--;
            if (!ptr->first)
                delete ptr;
        }
    }

    inline void clear() { 
        this->~fast_shared_ptr();
        ptr = nullptr;
     }

    inline fast_shared_ptr<T>& operator=(std::nullptr_t) {
        clear();
        return *this;
    }

    inline fast_shared_ptr<T>& operator=(T&& a) {
        this->~fast_shared_ptr();
        new(this) fast_shared_ptr<T>(move(a));
        return *this;
    }

    inline fast_shared_ptr<T>& operator=(const fast_shared_ptr<T>& other) {
        this->~fast_shared_ptr();
        new(this) fast_shared_ptr<T>(other);
        return *this;
    }

    inline operator bool() const { return ptr; }
    inline operator T() const { return ptr ? ptr->second : T(); }
    inline T& data() const { return ptr->second; }
    inline uint64 ref_cnt() const { return ptr ? ptr->first : 0; }
};

template<class T>
inline std::ostream& operator<<(std::ostream& os, const fast_shared_ptr<T>& p) {
    if (p)
        return os << p.ptr->second;
    return os << "null";
}


template<class T>
inline std::istream& operator>>(std::istream& is, fast_shared_ptr<T>& p) {
    T a;
    is >> a;
    p = move(a);
    return is;
}


template<class T>
struct Maybe {
    typedef T value_type;
    T* ptr;
    inline Maybe() { ptr = nullptr; }
    inline Maybe(std::nullptr_t) { ptr = nullptr; }
    inline Maybe(T* ptr) : ptr(ptr) {}
    inline operator bool() const { return ptr; }
};

} // jittor
