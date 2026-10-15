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
    // Non-owning metadata for the binding's GC traversal. Ownership remains
    // in the converter's deleter, exactly as for ordinary taped callbacks.
    void* python_callable = nullptr;
    inline ~GradCallback() { if (deleter) deleter(); }
    GradCallback(const GradCallback&) = delete;
    GradCallback() = default;
    GradCallback(GradCallback&& other) : func(other.func), deleter(other.deleter), python_callable(other.python_callable) {
        other.func = nullptr;
        other.deleter = nullptr;
        other.python_callable = nullptr;
    };
    GradCallback(std::function<void(int,Var**,int,VarPtr*)> && func, std::function<void()>&& deleter)
    : func(move(func)), deleter(move(deleter)) {};

    void operator =(GradCallback&& other) { this->~GradCallback(); new (this) GradCallback(move(other)); }
};

struct TapeOp final : Op {
    TapeOp(Var* x);
    
    const char* name() const override { return "tape"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    void run() override;
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
