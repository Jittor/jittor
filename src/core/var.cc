// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <type_traits>

#include "core/var.h"
#include "runtime/profiler/step_trace.h"
#include "core/op.h"
#include "core/grad.h"
#include "mem/allocator.h"
#include "bindings/pybind/py_var_tracer.h"
#include "mem/swap.h"
#include "runtime/device.h"
#include "ops/op_register.h"

namespace jittor {

std::atomic<int64> Var::number_of_lived_vars{0};

void Var::refresh_alias_requires_grad() {
    if (autograd_alias && autograd_alias->inherit_requires_grad)
        set_flag(VarFlags::_requires_grad_disabled,
                 !autograd_alias->requires_grad());
}

void Var::check_alias_history() const {
    USER_CHECK(!autograd_alias || !autograd_alias->stale_unrecorded_view())
        << "A view was created in no_grad mode and its base or another view "
           "of its base has been modified inplace. This use of the view is "
           "ambiguous; create the view and perform the inplace operation "
           "in the same grad mode.";
}

VarPtr contiguous_storage(Var* value) {
    if (value->is_contiguous()) return value;
    static auto make_contiguous = op_constructor<VarPtr, Var*>("contiguous");
    return make_contiguous(value);
}

VarPtr cast_operand_to_compute_dtype(Var* value, NanoString dtype) {
    // Only a floating operand can meet the preference, and only a floating
    // target is a compute dtype this cast can move to.
    if (!value->dtype().is_float() || !dtype.is_float()) return nullptr;
    if (value->dtype() == dtype) return nullptr;
    static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
    return make_unary(value, dtype);
}

DEFINE_FLAG(fast_shared_ptr<loop_options_t>, compile_options, {}, 
    "Override the default loop transfrom options");
DEFINE_FLAG(bool, no_grad, 0, 
    "No grad for all jittor Var creation");
DEFINE_FLAG(bool, no_fuse, 0, 
    "No fusion optimization for all jittor Var creation");
DEFINE_FLAG(uint8, node_order, 0, "id prior");
// TODO: fuse multiple flags
DEFINE_FLAG(int, amp_reg, 0, "Auto mixed-precision control registers, bit 0: prefer 32; bit 1: prefer 16; bit 2: keep reduce type; bit 3 keep white list type; bit 4: array like op prefer too; bit 5, reduce16 intermediate not use 32; bit 6: prefer bfloat16");

DEFINE_FLAG_WITH_SETTER(int, auto_mixed_precision_level, 0, "Auto mixed-precision optimization level, 0: not use fp16, 1-3: preserve level, not use fp16 for now; 4: perfer fp16, but some ops use fp32 e.g. sum,exp; 5: simular with 4, and array op will automatically convert to fp16; 6: all ops prefer fp16");

void (*_var_free_hook)(Var*);

// A var joins its parent's ring the moment the allocator agrees to share, and
// leaves it the moment it stops pointing into that allocation. A ring never
// holds a single var: the last two leave together.
// The rings are changed from two threads at once: the async worker allocates
// a batch's outputs -- linking a child into its parent's ring -- without the
// graph lock, while the Python thread frees vars, which unlinks them, under
// it. A link and an unlink in one ring rewrite the same neighbours' pointers,
// and the race left a dangling neighbour that a later unlink followed
// (signal 11 in share_group_unlink, from DeferredGraphWork::apply on the
// worker, about one pure-eager qwen3_decode_static run in three). Every ring
// change and walk holds this lock. It is only taken for vars that share an
// allocation, and is never held while taking another lock except the
// allocator's, by migrate_group.
std::mutex& share_ring_mutex() {
    static std::mutex* mutex = new std::mutex();
    return *mutex;
}

void share_group_link(Var* parent, Var* child) {
    std::lock_guard<std::mutex> guard(share_ring_mutex());
    if (PREDICT_BRANCH_NOT_TAKEN(child->share_next != nullptr))
        share_group_unlink_locked(child);
    if (!parent->share_next)
        parent->share_next = parent->share_prev = parent;
    child->share_prev = parent;
    child->share_next = parent->share_next;
    parent->share_next->share_prev = child;
    parent->share_next = child;
}

void share_group_unlink(Var* v) {
    // Only short bookkeeping is under this lock: the one operation that can
    // block for a long time -- the device copy in migrate_group -- runs
    // outside it (see mem/allocator.cc), so a waiter here is not held up by a
    // GIL-releasing transfer. The GIL must stay held: `Node::free` holds the
    // graph mutation lock across free_var, and dropping the GIL here would let
    // a Python thread that holds the GIL wait for that lock while this one
    // waits for the GIL back -- the inversion the readback probe hit.
    std::lock_guard<std::mutex> guard(share_ring_mutex());
    share_group_unlink_locked(v);
}

void share_group_unlink_locked(Var* v) {
    if (!v->share_next) return;
    Var* nxt = v->share_next;
    Var* prv = v->share_prev;
    v->share_next = v->share_prev = nullptr;
    if (nxt == v) return;
    prv->share_next = nxt;
    nxt->share_prev = prv;
    if (nxt->share_next == nxt)
        nxt->share_next = nxt->share_prev = nullptr;
}

void free_var(Var* v) {
    if (PREDICT_BRANCH_NOT_TAKEN((bool)_var_free_hook)) _var_free_hook(v);
    Var::number_of_lived_vars--;
    if (PREDICT_BRANCH_NOT_TAKEN(v->share_next != nullptr))
        share_group_unlink(v);
    if (save_mem)
        free_with_swap(v);
    else
    if (v->mem_ptr != nullptr) {
        auto mem_ptr = v->mem_ptr;
        auto allocation = v->allocation;
        auto allocator = v->allocator;
        v->mem_ptr = nullptr;
        v->allocator = nullptr;
        v->allocation = 0;
        allocator->free(mem_ptr, v->storage_span_bytes(), allocation);
    }
}

vector<Var*>* batch_released_vars = nullptr;

void free_var_mem(Var* v) {
    // See var.h. The executor asks "did this var's storage go while my batch
    // was running", and the only place that can answer is here.
    if (PREDICT_BRANCH_NOT_TAKEN(batch_released_vars != nullptr))
        batch_released_vars->push_back(v);
    if (PREDICT_BRANCH_NOT_TAKEN(v->share_next != nullptr))
        share_group_unlink(v);
    if (save_mem)
        free_with_swap(v);
    else {
        auto mem_ptr = v->mem_ptr;
        auto allocation = v->allocation;
        auto allocator = v->allocator;
        v->mem_ptr = nullptr;
        v->allocator = nullptr;
        v->allocation = 0;
        allocator->free(mem_ptr, v->storage_span_bytes(), allocation);
    }
}

void setter_auto_mixed_precision_level(const int& old_value, const int& value) {
    if (value <= 2) amp_reg = 0; else
    if (value == 3) amp_reg = amp_keep_reduce | amp_keep_white; else
    if (value == 4) amp_reg = amp_prefer16; else
    if (value == 5) amp_reg = amp_prefer16 | amp_array_prefer; else
    if (value == 6) amp_reg = amp_prefer16 | amp_array_prefer | amp_keep_reduce | amp_keep_white;
}

// Per-axis masks (reduce, broadcast, strided-input patterns) are 32-bit.
static inline void check_tensor_rank(const NanoVector& shape) {
    USER_CHECK(shape.size() <= NanoVector::max_tensor_rank)
        << "Jittor tensors support at most" << (int)NanoVector::max_tensor_rank
        << "dimensions, got shape" << shape;
}

Var::Var(NanoVector shape, NanoString dtype)
    : shape(shape), 
      loop_options(compile_options) {
    check_tensor_rank(shape);
    flags.set(NodeFlags::_var, 1);
    device_id = current_device();
    placement = current_tensor_placement();
    if (placement.explicit_backend)
        device_id = placement.device.backend == BackendId::Cpu ? -1 : placement.device.index;
    // complex dtypes are differentiable too (Wirtinger autograd), so they must not be
    // auto-stop_grad like integer/bool vars are. Only non-float AND non-complex stops grad.
    flags.set(NodeFlags::_stop_grad, (!dtype.is_float() && !dtype.is_complex()) || no_grad);
    set_flag(VarFlags::_stop_fuse, no_fuse);
    ns = dtype;
    ASSERT(ns.is_dtype());
    number_of_lived_vars++;
    numel();
    notify_node_created(this);
}
    
string Var::to_string() {
    string s = dtype().to_cstring();
    s += shape.to_string();
    return s;
}

int64 Var::numel() {
    bool negtive = 0;
    num=1;
    for (auto k : shape) {
        if (k<0) {
            negtive = 1;
            num *= -k;
        } else {
            num *= k;
        }
    }
    size = num * dsize();
    if (negtive) num = -num;
    return num;
}

void Var::set_shape(NanoVector shape) {
    check_tensor_rank(shape);
    this->shape = shape;
    storage_strides.clear();
    numel();
}

int64 Var::storage_stride(uint axis) const {
    if (storage_strides.size()) return storage_strides[axis];
    int64 stride = 1;
    for (uint i=axis+1; i<shape.size(); ++i) stride *= std::abs(shape[i]);
    return stride;
}

int64 Var::storage_span_bytes() const {
    if (!storage_strides.size() || !num) return size;
    int64 span = 1;
    for (uint i=0; i<shape.size(); ++i)
        span += (std::abs(shape[i])-1) * storage_strides[i];
    return span * dsize();
}

bool Var::is_contiguous() const {
    if (!storage_strides.size() || !num) return true;
    int64 expected = 1;
    for (int i=int(shape.size())-1; i>=0; --i) {
        if (shape[i] != 1 && storage_strides[i] != expected) return false;
        expected *= std::abs(shape[i]);
    }
    return true;
}

uint Var::stride_pattern() const {
    uint mask = 0;
    for (uint i=0; i<shape.size(); ++i)
        if (storage_stride(i) != 0) mask |= 1u << i;
    return mask;
}

// A view whose strides are exactly the contiguous ones needs no vector:
// `storage_stride` derives those from the shape, and the empty vector is
// already the "contiguous" sentinel every reader tests for. Storing them anyway
// is redundant and, past a point, costs a heap block -- the packed NanoVector
// holds 64 value bits, while the strides of a 5-D view of a large tensor (the
// video VAE decodes into `[1, 3, 28, 288, 512]`) need 75 and spill. Every
// value is identical either way, so dropping the vector changes no layout,
// only whether it is materialized.
static bool is_contiguous_strides(const NanoVector& shape, const int64* strides, int n) {
    int64 expected = 1;
    for (int i = n - 1; i >= 0; --i) {
        if (strides[i] != expected) return false;
        expected *= std::abs(shape[i]);
    }
    return true;
}

void Var::set_storage_strides(const vector<int64>& strides) {
    USER_CHECK((int)strides.size() == shape.size()) << "Storage stride rank must match shape";
    for (auto stride : strides)
        USER_CHECK(stride >= 0) << "Negative storage strides require an explicit offset view";
    if (is_contiguous_strides(shape, strides.data(), strides.size())) {
        storage_strides.clear();
        return;
    }
    storage_strides = NanoVector::make(strides.data(), strides.size());
}

void Var::set_storage_strides(NanoVector strides) {
    USER_CHECK(strides.size() == shape.size()) << "Storage stride rank must match shape";
    for (auto stride : strides)
        USER_CHECK(stride >= 0) << "Negative storage strides require an explicit offset view";
    vector<int64> expanded(strides.size());
    for (int i = 0; i < strides.size(); ++i) expanded[i] = strides[i];
    set_storage_strides(expanded);
}

bool Var::alloc(Allocator* allocator) {
    if (mem_ptr) return true;
    if (auto* x = share_src) {
        // x->allocator used to be dereferenced unconditionally: sharing with a
        // var that has not been allocated yet (nothing forces the source to be
        // allocated first) called a virtual function through a null pointer.
        // With the request in its own field the source's state can be asked
        // about, and an unusable source simply falls through to a real alloc.
        if (x->allocator && x->allocator->share_with(storage_span_bytes(), x->allocation, share_offset)) {
            mem_ptr = ((char*) x->mem_ptr) + share_offset;
            storage_offset_bytes = x->storage_offset_bytes + share_offset;
            allocation = x->allocation;
            this->allocator = x->allocator;
            // The request is consumed. This is what overwriting `allocator`
            // used to do implicitly; keeping it would re-alias on a later
            // alloc of the same var (free_var_mem then realloc).
            share_src = nullptr;
            share_offset = 0;
            // The request is gone but the *relationship* has to outlive it:
            // from here on this var is a sub-range of x's block and nothing
            // else records that, so migration would move it alone and drop
            // the alias. A parent with no block (a zero-sized var: see the
            // size==0 branch in the raw allocators) has no sub-ranges.
            if (mem_ptr) share_group_link(x, this);
            return true;
        }
        share_src = nullptr;
        share_offset = 0;
        USER_CHECK(size == 0 || (is_contiguous() && (!input() || !input()->is_storage_view())))
            << "Allocator cannot represent shared strided storage";
    }
    {
        StepTraceVarScope trace_var(this);
        mem_ptr = allocator->alloc(storage_span_bytes(), allocation);
    }
    storage_offset_bytes = 0;
    this->allocator = allocator;
    // A failed allocation throws (see AlignedAllocator::alloc and
    // CudaDeviceAllocator::alloc), so a null pointer here only means a
    // zero-sized var, which is allocated as far as callers are concerned.
    return mem_ptr != nullptr || size == 0;
}

VarPtr clone(Var* x);
void VarPtr::set_stop_grad(bool stop_grad) {
    if (stop_grad == ptr->is_stop_grad()) return;
    if (stop_grad)
        ptr->set_stop_grad();
    else {
        bool no_grad_bk = no_grad;
        AutogradPolicyOverride policy_guard({});
        no_grad = 0;
        *this = clone(ptr);
        no_grad = no_grad_bk;
    }
}

std::ostream& operator<<(std::ostream& os, const Var& var) {
    os << "Var" << '(' << var.id
        << ':' << var.liveness.forward.count()
        << ':' << var.liveness.backward.count()
        << ':' << var.liveness.pending.count()
        << ":i" << var._inputs.size()
        << ":o" << var._outputs.size()
        << ":s" << var.is_finished()
        << ":n" << var.flag(VarFlags::_needed_by_backward)
        << ":g" << !var.is_stop_grad()
        << ',' 
        << var.dtype().to_cstring() << ',' << var.name << ',' << std::hex <<(uint64)var.mem_ptr << std::dec
        << ')' << var.shape;
    if (trace_py_var) {
        os << '{';
        print_node_trace(&var, os);
        os << '}';
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Var* var) {
    return os << *var;
}
std::ostream& operator<<(std::ostream& os, const VarPtr& v) { return os << v.ptr; }

} // jittor
