// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
struct VarHolder;

struct Var : Node {
    NanoVector shape;
    cstr name;
    fast_shared_ptr<loop_options_t> loop_options;
    static int64 number_of_lived_vars;

    // this var will be generated after alloc.
    void* mem_ptr = nullptr;
    Allocator* allocator = nullptr;
    // Handle the allocator hands back from alloc(); its meaning belongs to
    // `allocator` and to nothing else. It used to have no initializer at all --
    // the only member of Var that had none -- so before alloc() it held stack
    // or heap residue, which the "already shared in place" checks in
    // getitem_op/setitem_op compare for equality.
    size_t allocation = 0;
    // Memory this var wants to alias, requested by share_with() and resolved by
    // alloc(). These two used to be stored in `allocator` and `allocation`, the
    // Var* reinterpreted as an Allocator*, told apart from a real allocator
    // only by mem_ptr == nullptr. So every `var->allocator->is_cuda()` that
    // could be reached between share_with() and alloc() was a virtual call on a
    // Var, and no caller could ask "is this var sharing?" without knowing that
    // unwritten rule.
    Var* share_src = nullptr;
    size_t share_offset = 0;
    int64 size, num;
    VarHolder* holder = nullptr;
    // Circular list of the vars that currently point into one allocation.
    // `share_src` above is only the *request*, and alloc() clears it once it
    // is served; from then on a child is indistinguishable from its parent
    // (same allocator, same allocation, a mem_ptr inside the parent's block),
    // so without this ring nothing can tell that moving one var's memory would
    // break another var's view of it -- see migrate_group in mem/allocator.cc.
    // Both null means "not shared"; a var is never alone in a ring.
    Var* share_prev = nullptr;
    Var* share_next = nullptr;
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
    int64 numel();
    void set_shape(NanoVector shape);
    bool alloc(Allocator* allocator);
    inline void share_with(Var* x, size_t offset = 0) { CHECK_EXIST; share_src = x; share_offset = offset; }
    // Whether alloc() still owes this var an aliased buffer.
    inline bool is_sharing() const { CHECK_EXIST; return share_src != nullptr; }
};

// Maintain the share ring described above. Linking happens once, in
// Var::alloc, when the underlying allocator accepts the share; unlinking
// happens wherever a var stops pointing at that allocation.
void share_group_link(Var* parent, Var* child);
void share_group_unlink(Var* v);

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
        if (ptr) {
            auto tmp = ptr;
            ptr = nullptr;
            tmp->release_both_liveness();
        }
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

    void set_stop_grad(bool stop_grad);
};

std::ostream& operator<<(std::ostream& os, const Var& var);
std::ostream& operator<<(std::ostream& os, const Var* var);
std::ostream& operator<<(std::ostream& os, const VarPtr& v);

} // jittor