// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <functional>
#include "core/op.h"
#include "core/var_holder.h"

namespace jittor {

struct Tapes;

struct GradCallback {
    typedef jittor::VarHolder VarHolder;
    typedef VarHolder* VarHolderPtr;
    typedef jittor::Var Var;
    typedef jittor::VarPtr VarPtr;
    std::function<void(int,Var**,int,VarPtr*)> func;
    std::function<void()> deleter;
    inline ~GradCallback() { if (deleter) deleter(); }
    GradCallback(const GradCallback&) = delete;
    GradCallback() = default;
    GradCallback(GradCallback&& other) : func(other.func), deleter(other.deleter) {
        other.func = nullptr;
        other.deleter = nullptr;
    };
    GradCallback(std::function<void(int,Var**,int,VarPtr*)> && func, std::function<void()>&& deleter)
    : func(move(func)), deleter(move(deleter)) {};

    void operator =(GradCallback&& other) { this->~GradCallback(); new (this) GradCallback(move(other)); }
};

struct TapeOp final : Op {
    TapeOp(Var* x);
    
    const char* name() const override { return "tape"; }
    // A tape marks a gradient boundary; it computes nothing, and `infer_shape`
    // gives its output the input's storage. Saying so is what every other
    // aliasing op here already does (clone, reshape, reinterpret_view), and it
    // skips the two things the runner would otherwise do for it: build a JIT
    // key for a kernel that does not exist, and call an empty `run`. Every
    // Python `Function` puts one tape on each differentiable input and each
    // output, so a transformer whose layer_norm and softmax take the fused
    // fast path runs 82 of these per forward pass.
    bool is_storage_view() const override { return true; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
};


struct Tapes final : Op {
    GradCallback callback;
    Tapes(
        const vector<VarHolder*>& taped_inputs,
        const vector<VarHolder*>& taped_outputs,
        GradCallback&& grad_callback
    );
    const char* name() const override { return "tapes"; }
    void grads(Var** douts, VarPtr* dins) override;
};


} // jittor