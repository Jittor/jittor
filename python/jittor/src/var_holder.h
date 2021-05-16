// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "var.h"
#include "ops/array_op.h"
#include "executor.h"
#include "mem/allocator/cuda_dual_allocator.h"

namespace jittor {

struct VarHolder;
VarPtr detach(Var* x);

struct DataView {
    VarHolder* vh;
    void* ptr;
    NanoVector shape;
    NanoString dtype;
};

struct ItemData {
    int64 data;
    NanoString dtype;
};

typedef struct _object PyObject;

// @pyjt(Var)
// @attrs(heaptype)
struct VarHolder {
    Var* var;
    list<VarHolder*>::iterator iter;
    VarHolder(Var* v);
    VarHolder(VarPtr&& v);
    // will move and delete v
    VarHolder(VarHolder* v);
    // @pyjt(__init__)
    VarHolder(PyObject* v, NanoString dtype=ns_void);
    // @pyjt(__dealloc__)
    ~VarHolder();
    string to_string();
    // @pyjt(sync)
    void sync(bool device_sync = false);
    // @pyjt(fetch_sync,numpy)
    ArrayArgs fetch_sync();

    /**
     * assign the data from another Var.
     */
    // @pyjt(assign)
    // @attrs(return_self)
    VarHolder* assign(VarHolder* v);

    /**
     * update parameter and global variable,
     * different from assign, it will
     * stop grad between origin var and assigned var, and
     * will update in the background
     */
    // @pyjt(update)
    // @attrs(return_self)
    VarHolder* update(VarHolder* v);

    /**
     * update parameter without set attribute.
     */
    // @pyjt(_update)
    // @attrs(return_self)
    VarHolder* _update(VarHolder* v);

    /**
     * swap the data with another Var.
     */ 
    // @pyjt(swap)
    // @attrs(return_self)
    inline VarHolder* swap(VarHolder* v) { std::swap(var, v->var); return this; };
    
    void operator=(VarPtr&& v);

    static list<VarHolder*> hold_vars;

    /** 
     * set the name of the Var.
     */
    // @pyjt(name)
    // @attrs(return_self)
    inline VarHolder* name(const char* s) {
        var->name = s;
        return this;
    }

    /** 
     * return the name of the Var.
     */
    // @pyjt(name)
    inline const char* name() {
        return var->name.c_str();
    }

    /** 
     * return the number of elements in the Var.
     */
    // @pyjt(numel)
    inline int64 numel() {
        if (var->num<0) sync();
        return var->num;
    }

    /** 
     * disable the gradient calculation for the Var.
     */
    // @pyjt(stop_grad)
    // @attrs(return_self)
    inline VarHolder* stop_grad() {
        var->set_stop_grad();
        return this;
    }

    /**
     * return True if the gradient is stopped.
     */
    // @pyjt(is_stop_grad)
    inline bool is_stop_grad() {
        return var->is_stop_grad();
    }

    /* detach the grad */
    // @pyjt(detach)
    inline VarHolder* detach() {
        return new VarHolder(jittor::detach(var));
    }


    /**
     * stop operator fusion.
     */
    // @pyjt(stop_fuse)
    // @attrs(return_self)
    inline VarHolder* stop_fuse() {
        var->flags.set(NodeFlags::_stop_fuse);
        return this;
    }

    /**
     * return True if operator fusion is stopped.
     */ 
    // @pyjt(is_stop_fuse)
    inline bool is_stop_fuse() {
        return var->flags.get(NodeFlags::_stop_fuse);
    }

    /** 
     * return the shape of the Var.
     */
    // @pyjt(__get__shape)
    inline NanoVector shape() {
        if (var->num<0) sync();
        return var->shape;
    }

    /** 
     * return True if the Var requires gradient calculation.
     * @see is_stop_grad
     */
    // @pyjt(__get__requires_grad)
    inline bool get_requires_grad() {
        return !var->is_stop_grad();
    }

    /**
     * enable or disable gradient calculation.
     * @see stop_grad
     */ 
    // @pyjt(__set__requires_grad)
    inline void set_requires_grad(bool flag) {
        if (flag == get_requires_grad()) return;
        if (flag)
            _update(this);
        else
            stop_grad(); 
        return;
    }

    // @pyjt(__get__uncertain_shape)
    inline NanoVector uncertain_shape() {
        return var->shape;
    }

    /**
     * return the data type of the Var.
     */
    // @pyjt(__get__dtype)
    inline NanoString dtype() {
        return var->dtype();
    }

    // @pyjt(__get__compile_options)
    inline loop_options_t compile_options() {
        return var->loop_options;
    }

    // @pyjt(__set__compile_options)
    inline void set_compile_options(loop_options_t&& options) {
        var->loop_options = move(options);
    }

    /**
     * get a numpy array which shares the data with the Var. 
     */
    // @pyjt(__get__data)
    inline DataView data() {
        sync(true);
        #ifdef HAS_CUDA
        migrate_to_cpu(var, exe.allocator);
        #endif
        return {this, var->mem_ptr, var->shape, var->dtype()};
    }

    /**
     * returns the Python number if the Var contains only one element.
     * For other cases, see data().
     */
    // @pyjt(item)
    ItemData item();

    /**
     * return the number of dimensions.
     */
    // @pyjt(__get__ndim)
    inline int ndim() {
        return var->shape.size();
    }

    // @pyjt(__set__data)
    inline void set_data(ArrayArgs&& array) {
        sync(true);
        ASSERT(array.dtype.dsize() == var->dtype().dsize()
            && array.dtype.is_int() == var->dtype().is_int());
        int64 size = array.dtype.dsize();
        for (int i=0; i<array.shape.size(); i++)
            size *= array.shape[i];
        ASSERT(size==var->size);
        #ifdef HAS_CUDA
        migrate_to_cpu(var, exe.allocator);
        #endif
        std::memcpy(var->mem_ptr, array.ptr, size);
    }

    // @pyjt(share_with)
    // @attrs(return_self)
    inline VarHolder* share_with(VarHolder* other) {
        CHECK(!var->allocator) << "This var is already executed or shared.";
        var->allocator = (Allocator*)(other->var);
        return this;
    }

    /**
     * print the information of the Var to debug.
     */
    // @pyjt(debug_msg)
    string debug_msg();
};

// @pyjt(sync)
void sync(const vector<VarHolder*>& vh=vector<VarHolder*>(), bool device_sync=false);
// @pyjt(fetch_sync)
vector<ArrayArgs> fetch_sync(const vector<VarHolder*>& vh);

// @pyjt(sync_all)
void sync_all(bool device_sync=false);

inline vector<Var*> convert(const vector<VarHolder*>& vhs) {
    vector<Var*> v;
    v.reserve(vhs.size());
    for (uint i=0; i<vhs.size(); i++) v.emplace_back(vhs[i]->var);
    return v;
}

inline vector<VarHolder*> make_vh_vector(vector<VarPtr>&& vps) {
    vector<VarHolder*> a;
    a.reserve(vps.size());
    for (auto& vp : vps)
        // a.emplace_back(move(vp));
        a.emplace_back(new VarHolder(move(vp)));
    return a;
}

} // jittor