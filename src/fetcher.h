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

struct FetchResult;

struct FetchFunc {
    typedef FetchResult R;
    std::function<void(R*)> callback;
    std::function<void()> deleter;
    FetchFunc() = default;
    FetchFunc(FetchFunc&& other) : callback(other.callback), deleter(other.deleter) {
        other.callback = nullptr;
        other.deleter = nullptr;
    };
    FetchFunc(const FetchFunc&) = delete;
    FetchFunc(std::function<void(R*)>&& callback) : callback(move(callback)) {}
    FetchFunc(std::function<void(R*)>&& callback, std::function<void()>&& deleter)
    : callback(move(callback)), deleter(move(deleter)) {};
    ~FetchFunc() {
        if (deleter) {
            deleter();
        }
    }
    void operator =(FetchFunc&& other) { this->~FetchFunc(); new (this) FetchFunc(move(other)); }
};

struct FetchResult {
    FetchFunc func;
    vector<Allocation> allocations;
    vector<ArrayArgs> arrays;

    inline void call() { func.callback(this); }
};

// @pyjt(fetch)
void fetch(const vector<VarHolder*>& vh, FetchFunc&& func);

} // jittor
