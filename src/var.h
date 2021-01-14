// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "node.h"
#include "misc/cstr.h"
#include "misc/fast_shared_ptr.h"

namespace jittor {

constexpr size_t alignment = 32;

struct Var : Node {
    NanoVector shape;
    cstr name;
    fast_shared_ptr<loop_options_t> loop_options;
    static int64_t number_of_lived_vars;

    // this var will be generated after alloc.
    void* mem_ptr = nullptr;
    Allocator* allocator = nullptr;
    size_t allocation;
    int64_t size, num;
    inline bool is_float() const { CHECK_EXIST; return ns.is_float(); }
    inline int dsize() const { CHECK_EXIST; return ns.dsize(); }
    inline NanoString dtype() const { CHECK_EXIST; return ns; }
    inline NanoString& dtype() { CHECK_EXIST; return ns; }
    template <typename T>
    inline T* ptr() { CHECK_EXIST; return (T*)mem_ptr; }
    inline Op* input() { CHECK_EXIST; return _inputs.size() ? (Op*)_inputs.front() : (Op*)nullptr; }
    inline Caster<Op*, Node::output_t> outputs()  { CHECK_EXIST; return &_outputs; }
    inline Caster<Node::var_output_t, Node::output_t> outputs_with_index() { CHECK_EXIST; return &_outputs; }
    inline Op* input(uint i) { return Node::input(i)->op(); }
    inline Op* output(uint i) { return Node::output(i)->op(); }

    Var(NanoVector shape, NanoString dtype);

    string to_string();
    int64_t numel();
    void set_shape(NanoVector shape);
    bool alloc(Allocator* allocator);
    inline void share_with(Var* x, size_t offset = 0) { CHECK_EXIST; allocator = (Allocator*)x; allocation = offset; }
};

struct VarPtr {
    Var* ptr;
    
    inline
    VarPtr(Var* ptr=nullptr) : ptr(ptr) {
        if (ptr) {
            ptr->own_both_liveness();
        }
    }
    
    inline
    VarPtr(VarPtr&& other) {
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    
    inline
    VarPtr(const VarPtr& other) : VarPtr(other.ptr) {
    }
    
    inline
    VarPtr(NanoVector shape, NanoString dtype) {
        ptr = new Var(shape, dtype);
        ptr->own_both_liveness();
    }
    
    inline
    ~VarPtr() { free_liveness(); }
    
    inline
    void free_liveness() {
        if (ptr)
            ptr->release_both_liveness();
    }
    
    inline Var* operator->() { return ptr; }
    inline operator Var*() { return ptr; }
    inline operator bool() { return ptr; }
    
    inline VarPtr& operator=(VarPtr&& other) {
        free_liveness();
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const Var& var);
std::ostream& operator<<(std::ostream& os, const Var* var);
std::ostream& operator<<(std::ostream& os, const VarPtr& v);

} // jittor