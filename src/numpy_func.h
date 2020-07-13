// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <functional>
#include "common.h"
#include "var_holder.h"
#include "ops/array_op.h"

namespace jittor {
    
struct NumpyResult;

struct NumpyFunc {
    typedef NumpyResult R;
    std::function<void(R*)> callback;
    std::function<void()> deleter;
    std::function<void()> inc_ref;
    NumpyFunc() = default;
    NumpyFunc(NumpyFunc&& other) : callback(other.callback), deleter(other.deleter), inc_ref(other.inc_ref) {
        other.callback = nullptr;
        other.deleter = nullptr;
        other.inc_ref = nullptr;
    };
    NumpyFunc(const NumpyFunc& other) : callback(other.callback), deleter(other.deleter), inc_ref(other.inc_ref) {
        inc_ref();
    };
    NumpyFunc(std::function<void(R*)>&& callback) : callback(move(callback)) {}
    NumpyFunc(std::function<void(R*)>&& callback, std::function<void()>&& deleter)
    : callback(move(callback)), deleter(move(deleter)) {};
    NumpyFunc(std::function<void(R*)>&& callback, std::function<void()>&& deleter, std::function<void()>&& inc_ref)
    : callback(move(callback)), deleter(move(deleter)), inc_ref(move(inc_ref)) {};
    ~NumpyFunc() {
        if (deleter) {
            deleter();
        }
    }
    void operator =(NumpyFunc&& other) { this->~NumpyFunc(); new (this) NumpyFunc(move(other)); }
};

struct NumpyResult {
    map<string, vector<DataView>> varrays;
    map<string, int> ints;
    map<string, DataView> arrays;
};

} // jittor