// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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
    NumpyFunc() = default;
    NumpyFunc(NumpyFunc&& other) : callback(other.callback), deleter(other.deleter) {
        other.callback = nullptr;
        other.deleter = nullptr;
    };
    NumpyFunc(const NumpyFunc&) = delete;
    NumpyFunc(std::function<void(R*)>&& callback) : callback(move(callback)) {}
    NumpyFunc(std::function<void(R*)>&& callback, std::function<void()>&& deleter)
    : callback(move(callback)), deleter(move(deleter)) {};
    ~NumpyFunc() {
        if (deleter) {
            deleter();
        }
    }
    void operator =(NumpyFunc&& other) { this->~NumpyFunc(); new (this) NumpyFunc(move(other)); }
};

struct NumpyResult {
    // vector<Allocation> allocations;
    map<string, vector<ArrayArgs>> varrays;
    map<string, int> ints;
    map<string, ArrayArgs> arrays;
    // mem ptr, dtype, shape --> numpy array
};

} // jittor