// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "var.h"
#include "ops/array_op.h"
#include "mem/allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"

namespace jittor {

struct VarHolder;
VarPtr detach(Var* x);
VarPtr device_copy(Var* x, int device);

struct DataView {
    VarHolder* vh;
    void* ptr;
    NanoVector shape;
    NanoString dtype;
};

/** The base object for the numpy array `Var.data` returns.
 *
 * It holds two claims, for two different reasons:
 *
 *  - an extra liveness on the Var that was current when `.data` was taken, so
 *    the allocation the view points at cannot be freed.  The base used to be
 *    the VarHolder's PyObject, which pins the python wrapper but not the
 *    memory: `assign` keeps the wrapper and swaps a *different* Var into it,
 *    releasing the old one.  `a = v.data; v.assign(other); a[0]` then read
 *    freed memory while its base object was still perfectly alive.
 *  - a reference to the VarHolder's PyObject, because `Var.data`'s documented
 *    lifetime is that the view keeps the whole Var alive
 *    (tests/core/test_array.py::TestArray::test_data asserts exactly that:
 *    `del a` must not change liveness_info() while the view is held).
 *
 * Returns a new reference, or nullptr with the python error set.
 */
PyObject* new_var_data_owner(VarHolder* vh);

struct ItemData {
    int64 data;
    NanoString dtype;
};

typedef struct _object PyObject;

EXTERN_LIB list<VarHolder*> hold_vars;
EXTERN_LIB list<VarHolder*>::iterator sync_ptr;

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
    // @attrs(return_self)
    VarHolder* sync(bool device_sync = false, bool weak_sync = true);

    /**
     * Returns a numpy array copy of the Var.
     */
    // @pyjt(fetch_sync,numpy)
    ArrayArgs fetch_sync();

    inline void release_holder() {var->holder = nullptr;}
    inline void own_holder() {var->holder = this;}

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
    inline VarHolder* swap(VarHolder* v) {
        std::swap(var, v->var);
        own_holder(); v->own_holder();
        return this; 
    };

    /**
     * The CUDA device index this Var lives on, or will be computed on; -1
     * when there is no CUDA device. Host residency is a different question
     * (see ``location``): a Var migrated to host memory keeps the device it
     * belongs to and goes back to it.
     */
    // @pyjt(__get__device_id)
    inline int device_id() { return var->device_id; }

    /**
     * Return this Var on CUDA device ``device``, copying it there when it
     * lives somewhere else -- the equivalent of torch's ``tensor.to("cuda:N")``.
     * Ops on the result run on that device, and gradients flow back to this
     * one. This is the only way data changes device.
     */
    // @pyjt(to_device)
    inline VarHolder* to_device(int device) {
        return new VarHolder(jittor::device_copy(var, device));
    }

    // @pyjt(location)
    inline string location() {
        if (var->flag(VarFlags::_is_swapped))
            return "disk";
        if (var->mem_ptr == nullptr)
            return "none";
        if (var->allocator->is_cuda())
            return "device";
        return "cpu";
    }

    // @pyjt(migrate_to_cpu)
    // @attrs(return_self)
    VarHolder* migrate_to_cpu_();

    // @pyjt(migrate_to_gpu)
    // @attrs(return_self)
    inline VarHolder* migrate_to_gpu_() {
        sync(true, false);
        #ifdef HAS_CUDA
        migrate_to_gpu(var, get_allocator());
        #endif
        return this;
    }
    
    void operator=(VarPtr&& v);


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
        return var->num;
    }

    /** 
     * return the number of bytes of this Var.
     */
    // @pyjt(__get__nbytes)
    inline int64 nbytes() {
        return var->num * var->dsize();
    }

    /** 
     * return id of this Var.
     */
    // @pyjt(__get__id)
    inline int64 id() {
        return var->id;
    }

    // @pyjt(__get__var_ptr)
    inline int64 var_ptr() {
        return (int64)var;
    }

    // @pyjt(__get__flags)
    inline int32 flags() {
        return (int32)(var->flags.flags);
    }

    /** 
     * disable the gradient calculation for the Var.
     */
    // @pyjt(stop_grad)
    // @attrs(return_self)
    inline VarHolder* stop_grad() {
        var->set_stop_grad();
        var->set_flag(VarFlags::_requires_grad_disabled, 0);
        var->set_flag(VarFlags::_first_order_only, 0);
        return this;
    }

    /**
     * return True if the gradient is stopped.
     */
    // @pyjt(is_stop_grad)
    inline bool is_stop_grad() {
        return var->is_stop_grad();
    }

    // @pyjt(_set_first_order_only)
    // @attrs(return_self)
    inline VarHolder* set_first_order_only() {
        var->set_flag(VarFlags::_first_order_only);
        return this;
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
        var->set_flag(VarFlags::_stop_fuse);
        return this;
    }

    /**
     * return True if operator fusion is stopped.
     */ 
    // @pyjt(is_stop_fuse)
    inline bool is_stop_fuse() {
        return var->flag(VarFlags::_stop_fuse);
    }

    /**
     * output hint for training optimization
     */
    // @pyjt(out_hint)
    // @attrs(return_self)
    inline VarHolder* out_hint() {
        var->set_flag(VarFlags::_out_hint);
        return this;
    }

    /** 
     * return the shape of the Var.
     */
    // @pyjt(__get__shape)
    inline NanoVector shape() {
        return var->shape;
    }

    // @pyjt(release_from_holders)
    void release_from_holders();

    /** 
     * return True if the Var requires gradient calculation.
     * @see is_stop_grad
     */
    // @pyjt(__get__requires_grad)
    inline bool get_requires_grad() {
        return !var->is_stop_grad()
            && !var->flag(VarFlags::_requires_grad_disabled);
    }

    /**
     * enable or disable gradient calculation.
     * @see stop_grad
     */ 
    // @pyjt(__set__requires_grad)
    void set_requires_grad(bool flag);

    /** 
     * enable the gradient calculation for the Var.
     */
    // @pyjt(start_grad)
    // @attrs(return_self)
    VarHolder* start_grad();

    /**
     * Whether this Var is a leaf of the backward graph: it requires gradient
     * and no differentiable predecessor can send it one. Always equal to
     * ``grad_fn_node_id == -1``.
     *
     * This is the graph fact ``torch.Tensor.is_leaf`` reports; the
     * compatibility layer is what maps torch's spelling onto it. See
     * ``jittor::backward_grad_fn`` in grad.h for the exact rule and for why
     * this costs the producer's arity rather than a graph walk.
     */
    // @pyjt(__get__is_backward_leaf)
    bool is_backward_leaf();

    /**
     * Identity of the op a gradient would flow into on its way to this Var, as
     * a ``Node`` id, or -1 when this Var is a backward leaf. Two Vars produced
     * by the same op share it; it is what a ``grad_fn`` object's identity
     * would be built on.
     */
    // @pyjt(__get__grad_fn_node_id)
    int64 grad_fn_node_id();

    /**
     * Which *kind* of op that is, as the registration-time operator id (see
     * ``Op::type_id``), or -1 when this Var is a backward leaf. 0 means the op
     * never reached the registry -- an out-of-tree op, or one a C++ unit test
     * built -- because a query must answer rather than raise.
     *
     * Callers deciding on operator identity compare this, never a name. 2.17
     * took the last string comparison of an operator's name out of the core and
     * this must not bring one back.
     */
    // @pyjt(__get__grad_fn_op_id)
    int64 grad_fn_op_id();

    /**
     * That op's name, for diagnostics only -- the fused spelling
     * (``binary.multiply``), and the empty string for a backward leaf.
     * Identity decisions belong to ``grad_fn_op_id``.
     */
    // @pyjt(__get__grad_fn_name)
    string grad_fn_name();

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
    DataView data();
    
    // @pyjt(__get__raw_ptr)
    uint64 raw_ptr();

    // @pyjt(__get__device_raw_ptr)
    inline uint64 device_raw_ptr() {
        sync(true, false);
        #ifdef HAS_CUDA
        if (!var->allocator->is_cuda())
            migrate_to_gpu(var, get_allocator());
        #endif
        return (uint64)var->mem_ptr;
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
    // @pyjt(__get__ndim, dim)
    inline int ndim() {
        return var->shape.size();
    }

    // @pyjt(__set__data)
    void set_data(ArrayArgs&& array);

    // @pyjt(share_with)
    // @attrs(return_self)
    inline VarHolder* share_with(VarHolder* other) {
        CHECK(!var->allocator && !var->is_sharing())
            << "This var is already executed or shared.";
        var->share_with(other->var);
        return this;
    }

    /**
     * print the information of the Var to debug.
     */
    // @pyjt(debug_msg)
    string debug_msg();

    /* Jittor Var doesn't have this interface, please change your code as below::

    model = Model()
    optimizer = SGD(model.parameters())
    ...
    optimizer.backward(loss)
    
    for p in model.parameters():
        # prev code:
        # grad = p.grad

        # change to:
        grad = p.opt_grad(optimizer)
     */
    // @pyjt(__get__grad)
    int grad();

    // @pyjt(_input)
    inline VarHolder* _input(int i) {
        CHECK(!var->is_finished());
        return new VarHolder(var->input()->input(i));
    }

    /* Add dependency, make var computed after vars
    */
    // @pyjt(_add_dependency)
    // @attrs(return_self)
    inline VarHolder* _add_dependency(vector<VarHolder*>&& vars) {
        vector<Node*> b(vars.size());
        for (int i=0; i<vars.size(); i++)
            b[i] = vars[i]->var;
        CHECK(!var->is_finished());
        auto a = var->input();
        var->input()->add_inputs(b);
        auto edge = a->_inputs.end();
        for (int i=0; i<b.size(); i++) {
            edge = std::prev(edge);
            // set -1 mean this is a control dependency edge
            edge->reverse().index = -1;
        }
        return this;
    }

    /* check a[x][y] = c
    */
    // @pyjt(check_cascade_setitem)
    // @attrs(return_self)
    VarHolder* check_cascade_setitem(VarHolder* out);
};

// @pyjt(sync)
void sync(const vector<VarHolder*>& vh=vector<VarHolder*>(), bool device_sync=false, bool weak_sync=true);
// @pyjt(fetch_sync)
vector<ArrayArgs> fetch_sync(const vector<VarHolder*>& vh);

// @pyjt(sync_all)
void sync_all(bool device_sync=false);

// Called after a complete VarHolder has crossed into a Python object.
void schedule_pending_from_python(VarHolder* holder);

// Start this Var's pending subgraph without a device synchronization.
// @pyjt(submit_pending)
void submit_pending(VarHolder* holder);

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

// @pyjt(ternary_out_hint)
VarHolder* ternary_out_hint(VarHolder* cond, VarHolder* x, VarHolder* y);

// @pyjt(migrate_all_to_cpu)
void migrate_all_to_cpu();

// @pyjt(wrap_var_addr)
inline VarHolder* wrap_var_addr(int64 addr) {
    return new VarHolder((Var*)addr);
}

// @pyjt(reuse_np_array)
VarHolder* reuse_np_array(PyObject* obj);

} // jittor
