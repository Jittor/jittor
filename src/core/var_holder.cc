// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include "core/var_holder.h"
#include "ops/layout_propagation.h"
#include "core/var.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "runtime/backend.h"
#include "core/graph.h"
#include "core/grad.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "ops/op_register.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/fuse_transpose_op.h"
#include "ops/composite/reshape_op.h"
#include "ops/composite/clone_op.h"
#include "type/fp16_compute.h"
#include "mem/swap.h"
#include "runtime/executor_entry.h"
#include "runtime/holder_state.h"
#include "bindings/pyjt/py_converter.h"

namespace jittor {

namespace {
struct VarDataOwner {
    PyObject* holder;
    Var* var;
    Allocation allocation;
};
}

static void free_var_data_owner(PyObject* capsule) {
    auto owner = (VarDataOwner*)PyCapsule_GetPointer(capsule, "jittor.var_data");
    if (!owner) return;
    owner->var->release_both_liveness();
    Py_XDECREF(owner->holder);
    delete owner;
}

PyObject* new_var_data_owner(VarHolder* vh) {
    auto owner = std::unique_ptr<VarDataOwner>(
        new VarDataOwner{GET_OBJ_FROM_RAW_PTR(vh), vh->var});
    auto* var = owner->var;
    // Keeping the Var alive is not enough: a later CUDA consumer can migrate
    // it back to the device and release the host block exported by .data.
    // Hold that exact allocation independently of the Var's current storage.
    if (var->mem_ptr) {
        CHECK(var->allocator->share_with(var->storage_span_bytes(),
            var->allocation, var->storage_offset_bytes))
            << "Cannot retain storage exported by Var.data";
        owner->allocation.ptr = var->mem_ptr;
        owner->allocation.allocation = var->allocation;
        owner->allocation.size = var->storage_span_bytes();
        owner->allocation.allocator = var->allocator;
    }
    owner->var->own_both_liveness();
    Py_INCREF(owner->holder);
    auto capsule = PyCapsule_New((void*)owner.get(), "jittor.var_data",
        &free_var_data_owner);
    if (!capsule) {
        owner->var->release_both_liveness();
        Py_DECREF(owner->holder);
        return nullptr;
    }
    owner.release();
    return capsule;
}

// Holders rebound while a step is being captured, with the Var each held
// first. See `state_capture_begin`.
static std::unordered_map<VarHolder*, VarPtr>* state_capture = nullptr;

static inline void note_rebind(VarHolder* holder) {
    if (PREDICT_BRANCH_NOT_TAKEN(state_capture != nullptr)
            && !state_capture->count(holder))
        state_capture->emplace(holder, VarPtr(holder->var));
}

// A capture's record stops counting once its holder holds something else.
static inline void drop_capture_record(VarHolder* holder) {
    if (PREDICT_BRANCH_NOT_TAKEN(holder->capture_record)) {
        holder->capture_record = false;
        holder->var->capture_owners--;
    }
}

static bool capture_saw_readback = false;

void state_capture_begin() {
    USER_CHECK(!state_capture) << "a step capture is already recording state";
    state_capture = new std::unordered_map<VarHolder*, VarPtr>();
    capture_saw_readback = false;
}

PyObject* state_capture_end() {
    unique_ptr<std::unordered_map<VarHolder*, VarPtr>> records(state_capture);
    state_capture = nullptr;
    PyObjHolder result(PyList_New(0));
    if (!records) return result.release();
    for (auto& kv : *records) {
        VarHolder* holder = kv.first;
        Var* old = kv.second.ptr;
        // Rebound and back again, or state created inside the capture: a
        // temporary's first Var is not an executed leaf.
        if (holder->var == old || !old->is_finished() || !old->mem_ptr) continue;
        PyObject* holder_obj = GET_OBJ_FROM_RAW_PTR(holder);
        Py_INCREF(holder_obj);
        PyObjHolder item(PyTuple_New(3));
        PyTuple_SET_ITEM(item.obj, 0, holder_obj);
        auto* record = new VarHolder(old);
        record->capture_record = true;
        ASSERT(old->capture_owners < UINT16_MAX);
        old->capture_owners++;
        PyTuple_SET_ITEM(item.obj, 1, to_py_object<VarHolder*>(record));
        PyTuple_SET_ITEM(item.obj, 2, to_py_object<VarHolder*>(new VarHolder(holder->var)));
        if (PyList_Append(result.obj, item.obj) < 0)
            throw std::runtime_error("cannot build the step capture's state list");
    }
    return result.release();
}

void add_hold_vars(VarHolder* self) {
    self->iter = runtime_holder_state().add(self);
}

void schedule_pending_from_python(VarHolder* holder) {
    if (holder->var->placement.explicit_backend)
        holder->var->set_flag(VarFlags::_placement_published);
    runtime_executor().submit_pending(holder->var);
}

void submit_pending(VarHolder* holder) {
    runtime_executor().submit_pending(holder->var, true);
}

// Everything below that pairs `sync()` with `migrate_to_cpu()` takes the
// executor lock across the pair rather than letting `sync()` take it and drop
// it again. The migration allocates from and frees into the same pools a batch
// uses, and it is where the device-to-host transfer releases the GIL
// (`DeviceWaitScope` in mem/allocator.cc) -- which it only does while this lock
// is held, so holding it here is what makes that release reachable at all.
VarHolder* VarHolder::migrate_to_cpu_() {
    ExecutorEntryScope entry;
    sync(true, false);
#ifdef HAS_ACCELERATOR
    migrate_to_cpu(var, runtime_executor().allocator);
#endif
    return this;
}

// Every read of a Var's value by the host. A capture asks whether the call it
// traced read one: its Python path then depends on tensor values, and a
// replay, which does not run that Python, would repeat whatever it did the
// first time.
static int64 host_readbacks = 0;
int64 host_readback_count() { return host_readbacks; }

int first_finished(const vector<VarHolder*>& vars) {
    for (int i = 0; i < (int)vars.size(); i++)
        if (vars[i]->var->is_finished()) return i;
    return -1;
}

int first_rebound(const vector<VarHolder*>& holders, const vector<VarHolder*>& olds) {
    CHECK(holders.size() == olds.size());
    for (int i = 0; i < (int)holders.size(); i++)
        if (holders[i]->var != olds[i]->var) return i;
    return -1;
}

DECLARE_FLAG(int, keep_graph);
#ifdef HAS_ACCELERATOR
DECLARE_FLAG(int, use_cuda_managed_allocator);
#endif

// A read inside a captured step makes the capture unreplayable, and the step
// has to happen exactly once all the same. So from the first read on, the
// step runs as written: nothing more is kept, and what was kept so far --
// which the read is about to execute -- is let go, so that it finishes here
// rather than running again when the refused capture is finished off.
static inline void note_readback() {
    host_readbacks++;
    if (PREDICT_BRANCH_NOT_TAKEN(state_capture != nullptr) && !capture_saw_readback) {
        capture_saw_readback = true;
        keep_graph = 0;
        for (auto* holder : runtime_holder_state().holders())
            if (holder->var && !holder->var->is_finished()) holder->release_kept();
    }
}

// Whether a read of a Var's value has to wait for the devices, beyond its
// own producer. It does not: device memory is read back by a copy that waits
// for the stream that produced it, and host memory is written by host ops.
// Only managed memory is read in place, where a device may still be writing.
// Waiting for every device made each `.numpy()` of a host tensor -- a label,
// a scheduler's timestep -- wait for all the work queued before it.
static inline bool readback_waits_for_devices() {
#ifdef HAS_ACCELERATOR
    return use_cuda_managed_allocator;
#else
    return false;
#endif
}

// Whether the value can be read from host memory without running anything.
static inline bool host_value_ready(Var* var) {
    return var->mem_ptr && !var->allocator->is_cuda()
        && !var->flag(VarFlags::_execution_failed);
}

DataView VarHolder::data() {
    note_readback();
    if (!host_value_ready(var)) {
        ExecutorEntryScope entry;
        sync(readback_waits_for_devices(), false);
        var->set_flag(VarFlags::_execution_failed, 0);
#ifdef HAS_ACCELERATOR
        migrate_to_cpu(var, runtime_executor().allocator);
#endif
    }
    return {this, var->mem_ptr, var->shape, var->dtype(), var->storage_strides};
}

uint64 VarHolder::raw_ptr() {
    note_readback();
    ExecutorEntryScope entry;
    sync(true, false);
#ifdef HAS_ACCELERATOR
    migrate_to_cpu(var, runtime_executor().allocator);
#endif
    return (uint64)var->mem_ptr;
}

// Both Vars, described the same way, so the two entry points refuse the same
// things in the same words.
static void check_inplace_target(Var* var, const char* what) {
    USER_CHECK(var->mem_ptr) << what << "needs an allocated tensor";
    USER_CHECK(var->is_contiguous())
        << what << "needs a dense tensor; got strides"
        << var->storage_strides << "for shape" << var->shape;
}

void VarHolder::write_inplace(ArrayArgs&& array) {
    ExecutorEntryScope entry;
    // Not a device sync. The copy below goes to the same device as the
    // buffer and is ordered against the work already queued on it, so the
    // host does not have to stand and wait for the previous step to drain --
    // which is the whole point when this is feeding a kept graph once per
    // step. `sync(false, ...)` still resolves anything this Var is waiting on.
    sync(false, false);
    check_inplace_target(var, "_write_inplace");
    USER_CHECK(array.dtype.dsize() == var->dtype().dsize()
        && array.dtype.is_int() == var->dtype().is_int())
        << "_write_inplace dtype mismatch:" << array.dtype << "into" << var->dtype();
    int64 size = array.dtype.dsize();
    for (int i=0; i<array.shape.size(); i++)
        size *= array.shape[i];
    USER_CHECK(size == var->size)
        << "_write_inplace size mismatch:" << size << "bytes into" << var->size;
    // Wherever the buffer already lives; no migration either way.
    Device dst{};
    if (var->allocator && var->allocator->is_cuda())
        dst = Device{accelerator_backend_id(), var->device_id < 0 ? 0 : var->device_id};
    // Ordered, not blocking: the bytes have to land before the kernels that
    // read them, which is stream order, not a host-side wait. See the H2D
    // branch in the CUDA backend's copy() for why the difference is 1000x.
    backend_copy(var->mem_ptr, dst, array.ptr, {}, size, true);
}

enum class StorageCopy { Direct, Snapshot, NoOp };

static StorageCopy validate_storage_copy(Var* var, Var* source) {
    USER_CHECK(source->dtype() == var->dtype())
        << "_copy_into dtype mismatch:" << source->dtype() << "into" << var->dtype();
    USER_CHECK(source->size == var->size)
        << "_copy_into size mismatch:" << source->size << "bytes into" << var->size;
    if (var->size == 0) return StorageCopy::NoOp;
    USER_CHECK(source->mem_ptr) << "_copy_into needs an allocated source";
    USER_CHECK(var->mem_ptr) << "_copy_into needs an allocated tensor";
    // Basic positive-stride slices and permutations have disjoint elements.
    // Validate that before issuing any write; expanded/overlapping targets
    // cannot be updated by scattering a dense source into their storage.
    auto ordered_dimensions = [](Var* value) {
        vector<std::pair<int64, int64>> dimensions;
        for (uint d=0; d<value->shape.size(); ++d)
            if (value->shape[d] > 1)
                dimensions.emplace_back(value->storage_stride(d), value->shape[d]);
        std::sort(dimensions.begin(), dimensions.end());
        return dimensions;
    };
    int64 span = 1;
    bool target_dense = true;
    if (!var->is_contiguous())
        for (auto dimension : ordered_dimensions(var)) {
            USER_CHECK(dimension.first >= span)
                << "_copy_into needs non-overlapping storage; got strides"
                << var->storage_strides << "for shape" << var->shape;
            target_dense = target_dense && dimension.first == span;
            span += (dimension.second-1) * dimension.first;
        }
    bool overlap = false;
    if (source->allocator == var->allocator
            && source->allocation == var->allocation) {
        bool same_layout = source->shape == var->shape;
        if (same_layout)
            for (uint d=0; d<var->shape.size(); ++d)
                if (var->shape[d] > 1
                        && source->storage_stride(d) != var->storage_stride(d))
                    same_layout = false;
        // Identical storage/layout is a no-op, including a strided self-copy.
        if (source->mem_ptr == var->mem_ptr && (same_layout
                || (source->is_contiguous() && var->is_contiguous()))) return StorageCopy::NoOp;
        auto dst_address = reinterpret_cast<uintptr_t>(var->mem_ptr);
        auto src_address = reinterpret_cast<uintptr_t>(source->mem_ptr);
        overlap = dst_address < src_address + source->storage_span_bytes()
            && src_address < dst_address + var->storage_span_bytes();
        int64 source_span = 1;
        bool source_dense = true;
        if (!source->is_contiguous())
            for (auto dimension : ordered_dimensions(source)) {
                source_dense = source_dense && dimension.first == source_span;
                source_span *= dimension.second;
            }
        USER_CHECK(!overlap || !target_dense || !source_dense)
            << "_copy_into source and destination partially overlap";
        if (overlap) {
            // Two gapped views can have overlapping address spans while
            // selecting disjoint elements (even/odd columns, for example).
            // Different residues modulo every stride's common divisor prove
            // disjointness without walking tensor data or allocating an
            // element-address set. Refuse aliases this cannot prove safe.
            int64 divisor = 0;
            for (Var* value : {var, source})
                for (uint d=0; d<value->shape.size(); ++d)
                    if (value->shape[d] > 1) {
                        int64 stride = value->storage_stride(d);
                        while (stride) {
                            int64 remainder = divisor % stride;
                            divisor = stride;
                            stride = remainder;
                        }
                    }
            auto separation = dst_address >= src_address
                ? dst_address-src_address : src_address-dst_address;
            USER_CHECK(divisor && separation % (divisor * var->dsize()))
                << "_copy_into source and destination overlap or cannot be proven disjoint";
        }
    }
    return overlap ? StorageCopy::Snapshot : StorageCopy::Direct;
}

static void validate_logical_copy_alias(VarHolder* target, VarHolder* source) {
    if (target->var == source->var) return;
    auto* target_root = target->view && target->view->base ? target->view->base : target;
    auto* source_root = source->view && source->view->base ? source->view->base : source;
    if (target_root != source_root) return;
    USER_CHECK((!target->view || target->storage_view_is_physical())
            && (!source->view || source->storage_view_is_physical()))
        << "_copy_into source and destination have overlapping logical view storage";
}

void VarHolder::copy_into(VarHolder* src, bool sync_src) {
    check_inplace_autograd();
    ExecutorEntryScope entry;
    // Neither sync waits on the device; both only resolve what the Var is
    // still pending on. The copy itself is stream-ordered against the work
    // already queued, which is what makes this usable once per step.
    sync(false, false);
    // A kept graph re-executes whenever its output is synced, so a reader
    // that already has the bytes -- the caller of a replayed graph -- must be
    // able to say so instead of paying for the whole graph again.
    if (sync_src) src->sync(false, false);
    USER_CHECK(src->var->mem_ptr || src->var->size == 0)
        << "_copy_into(sync_src=False) needs a source that already holds its "
           "bytes; this one has never been executed";
    validate_logical_copy_alias(this, src);
    auto plan = validate_storage_copy(var, src->var);
    record_inplace_write();
    if (plan == StorageCopy::NoOp) return;
    // A strided source -- a channels-last activation read as NCHW, a
    // transposed view -- is made dense first; the copy below moves raw bytes.
    unique_ptr<VarHolder> dense;
    if (plan == StorageCopy::Snapshot) {
        // Proven disjoint gapped aliases may interleave within their spans.
        // Snapshot via the existing device copy op before any destination
        // write, rather than reading a source that earlier runs overwrite.
        static auto make_copy = op_constructor<VarPtr, Var*>("copy");
        dense.reset(new VarHolder(make_copy(src->var)));
        dense->sync(false, false);
        src = dense.get();
    } else if (!src->var->is_contiguous()) {
        dense.reset(new VarHolder(contiguous_storage(src->var)));
        dense->sync(false, false);
        src = dense.get();
    }
    check_inplace_target(src->var, "_copy_into source");
    // Already there: an in-place update wrote into this very buffer (a step
    // capture taking over its state, see `capture_owners`).
    if (src->var->mem_ptr == var->mem_ptr) return;
    if (src->var->allocator == var->allocator
            && src->var->allocation == var->allocation) {
        auto dst_address = reinterpret_cast<uintptr_t>(var->mem_ptr);
        auto src_address = reinterpret_cast<uintptr_t>(src->var->mem_ptr);
        USER_CHECK(dst_address + var->size <= src_address
                || src_address + src->var->size <= dst_address)
            << "_copy_into source and destination partially overlap";
    }
    auto device_of = [](Var* v) {
        Device d{};
        if (v->allocator && v->allocator->is_cuda())
            d = Device{accelerator_backend_id(), v->device_id < 0 ? 0 : v->device_id};
        return d;
    };
    auto destination_device = device_of(var);
    auto source_device = device_of(src->var);
    if (var->is_contiguous()) {
        backend_copy(var->mem_ptr, destination_device,
                     src->var->mem_ptr, source_device, var->size, true);
        return;
    }
    // Coalesce the innermost contiguous dimensions. Each run uses the same
    // backend/stream copy as the dense path, without rebinding either holder
    // or reading accelerator data on the host.
    int64 run = 1;
    int leading = var->shape.size();
    for (int d=int(var->shape.size())-1; d>=0; --d) {
        if (var->shape[d] != 1 && var->storage_stride(d) != run) break;
        run *= var->shape[d];
        leading = d;
    }
    auto width = var->dsize();
    for (int64 i=0; i<var->num; i+=run) {
        int64 remainder = i/run, offset = 0;
        for (int d=leading-1; d>=0; --d) {
            offset += (remainder % var->shape[d]) * var->storage_stride(d);
            remainder /= var->shape[d];
        }
        backend_copy(static_cast<char*>(var->mem_ptr)+offset*width, destination_device,
                     static_cast<char*>(src->var->mem_ptr)+i*width, source_device,
                     run*width, true);
    }
}

void VarHolder::release_kept() {
    ExecutorEntryScope entry;
    // Breadth is irrelevant, reaching everything is not: an intermediate is
    // only reclaimable once nothing upstream of the output still carries the
    // mark, and the graph is a DAG, so the visited set is what keeps a
    // diamond from being walked twice.
    vector<Var*> stack{var};
    std::unordered_set<Var*> seen{var};
    while (stack.size()) {
        Var* v = stack.back();
        stack.pop_back();
        v->set_flag(VarFlags::_kept, 0);
        Op* op = v->input();
        if (!op) continue;
        for (Var* in : op->inputs())
            if (seen.insert(in).second)
                stack.push_back(in);
    }
}

void VarHolder::set_data(ArrayArgs&& array) {
    ExecutorEntryScope entry;
    sync(true);
    for (uint i=0; i<var->shape.size(); ++i)
        USER_CHECK(var->shape[i] <= 1 || var->storage_stride(i) != 0)
            << "Dense data assignment requires non-overlapping storage";
    USER_CHECK(array.dtype.dsize() == var->dtype().dsize()
        && array.dtype.is_int() == var->dtype().is_int());
    int64 size = array.dtype.dsize();
    for (int i=0; i<array.shape.size(); i++)
        size *= array.shape[i];
    USER_CHECK(size==var->size);
#ifdef HAS_ACCELERATOR
    migrate_to_cpu(var, runtime_executor().allocator);
#endif
    if (var->is_contiguous()) {
        std::memcpy(var->mem_ptr, array.ptr, size);
    } else {
        for (int64 i=0; i<var->num; ++i) {
            int64 remainder=i, offset=0;
            for (int d=int(var->shape.size())-1; d>=0; --d) {
                offset += (remainder % var->shape[d]) * var->storage_stride(d);
                remainder /= var->shape[d];
            }
            std::memcpy((char*)var->mem_ptr+offset*var->dsize(),
                        (const char*)array.ptr+i*var->dsize(), var->dsize());
        }
    }
}

VarHolder::VarHolder(Var* v) : var(v) {
    // Var holder has both forward and backward liveness
    own_holder();
    var->own_both_liveness();
    add_hold_vars(this);
}

VarHolder::VarHolder(VarPtr&& v) {
    var = v.ptr;
    v.ptr = nullptr;
    own_holder();
    add_hold_vars(this);
}

VarHolder::VarHolder(VarHolder* v) : var(v->var) {
    own_holder();
    iter = v->iter;
    *iter = this;
    capture_record = v->capture_record;
    // `v` is discarded without running ~VarHolder, so what ~VarHolder would
    // have relinked has to be relinked here: `v`'s own view record and the
    // records of every view that named `v` as its base. The records themselves
    // do not move, so only the back-pointers on the other side change.
    view = v->view;
    if (view) view->owner = this;
    views = v->views;
    for (auto* w = views; w; w = w->next) w->base = this;
    v->view = nullptr;
    v->views = nullptr;
    // free memory without calling deconstructor
    operator delete(v);
}

// The runtime owner maintains the weak-sync cursor and makes repeated unlink
// safe, including destruction after release_from_holders().
static inline void unlink_from_hold_vars(list<VarHolder*>::iterator& iter) {
    runtime_holder_state().erase(iter);
}

void VarHolder::release_from_holders() {
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (runtime_holder_state().contains(iter)) {
        unlink_from_hold_vars(iter);
        release_holder();
    }
}

static auto make_array_from_pyobj = op_constructor<VarPtr, PyObject*>("array");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_setitem = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");
static auto make_getitem = op_constructor<VarPtr, Var*, VarSlices&&>("getitem");
static auto make_transpose_view = op_constructor<VarPtr, Var*, NanoVector>("transpose");

static auto make_storage_reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");
static auto make_storage_broadcast = op_constructor<VarPtr, Var*, NanoVector, NanoVector>("broadcast_to");
static VarPtr apply_view_step(Var* value, const VarViewStep& step,
                             bool physical = false) {
    if (step.kind == VarViewStep::Expand)
        return make_storage_broadcast(value, NanoVector(step.axes), NanoVector());
    if (step.kind == VarViewStep::Reshape)
        return make_storage_reshape(value, NanoVector(step.axes));
    if (step.kind == VarViewStep::Transpose)
        return physical ? storage_view_transpose(value, NanoVector(step.axes))
                        : make_transpose_view(value, NanoVector(step.axes));
    return make_getitem(value, VarSlices(step.slices));
}

VarHolder::VarHolder(PyObject* obj, NanoString dtype) {
    auto vp = make_array_from_pyobj(obj);
    if (dtype != ns_void)
        vp = make_unary(vp, dtype);
    var = vp.ptr;
    vp.ptr = nullptr;
    own_holder();
    add_hold_vars(this);
}


static void unlink_view_owner(VarView* view) {
    if (auto* base = view->base) {
        if (view->prev) view->prev->next = view->next;
        else base->views = view->next;
        if (view->next) view->next->prev = view->prev;
    }
    view->base = nullptr;
    view->prev = view->next = nullptr;
}

VarView::~VarView() { Py_XDECREF(base_owner); }

void VarHolder::bind_python_owner(PyObject* owner, bool retain_view_base) {
    python_owner = owner;
    retain_view_base_owner = retain_view_base;
    retain_python_view_base();
}

void VarHolder::retain_python_view_base() {
    if (!retain_view_base_owner || !view || view->base_owner || !view->base)
        return;
    auto* root = view->base;
    // A .data replacement can orphan a root's writeback while its public
    // provenance still names the original root object.
    PyObject* owner = root->view && root->view->base_owner
        ? root->view->base_owner : root->python_owner;
    if (!owner || owner == python_owner) return;
    Py_INCREF(owner);
    view->base_owner = owner;
}

PyObject* VarHolder::collectable_view_base() {
    return view ? view->base_owner : nullptr;
}

void VarHolder::clear_collectable_view_base() {
    if (view) Py_CLEAR(view->base_owner);
}

PyObject* VarHolder::view_base_object() {
    PyObject* owner = collectable_view_base();
    if (!owner && view && view->base) owner = view->base->python_owner;
    if (!owner) owner = Py_None;
    Py_INCREF(owner);
    return owner;
}

void VarHolder::drop_view() {
    if (!view) return;
    auto* previous = view;
    view = nullptr;
    unlink_view_owner(previous);
    // Releasing a retained Python root can run user finalizers. They must
    // observe the detached record, never a record currently being destroyed.
    delete previous;
}

static bool shares_physical_storage(Var* value, Var* base) {
    // A pending share_with link is positive evidence before allocation. Two
    // null pointers are not; after execution allocation identity is decisive.
    Var* slow = value;
    Var* fast = value;
    while (value) {
        if (value == base) return true;
        if (value->mem_ptr && base->mem_ptr && value->allocator
                && value->allocator == base->allocator
                && value->allocation == base->allocation)
            return true;
        value = value->share_src;
        slow = slow ? slow->share_src : nullptr;
        fast = fast && fast->share_src ? fast->share_src->share_src : nullptr;
        // Malformed sharing cycles cannot establish a view relationship.
        if (slow && slow == fast) return false;
    }
    return false;
}

bool VarHolder::storage_view_is_physical() {
    if (!view || !view->physical_storage) return false;
    return !view->base || shares_physical_storage(var, view->base->var);
}

bool VarHolder::view_base_is_trainable_leaf() {
    if (!view) return false;
    auto* owner = this;
    while (owner->view && owner->view->base)
        owner = owner->view->base;
    Var* base = owner->var;
    if (owner->view) {
        // A Python owner may expire while the differentiable view still owns
        // its native graph. Follow only actual view producers to recover the
        // base; a clone/detach or an arithmetic producer is a real boundary.
        while (Op* producer = base->input()) {
            // Tensor(source) has a CloneOp producer but is a real view only
            // when native view metadata records that provenance. Ordinary
            // clone/detach must remain independent boundaries.
            bool constructor_view = dynamic_cast<CloneOp*>(producer)
                && base->autograd_alias && base->autograd_alias->source
                && base->autograd_alias->recorded_view_history;
            if (producer->is_stop_grad()
                    || !(producer->is_op(op_ids::getitem())
                        || dynamic_cast<ReshapeOp*>(producer)
                        || producer->is_op(op_ids::transpose())
                        || dynamic_cast<FuseTransposeOp*>(producer)
                        || producer->is_op(op_ids::broadcast_to())
                        || constructor_view)
                    || producer->_inputs.empty())
                break;
            Var* next = producer->_inputs.front().node->var();
            if (next == base) return false;
            base = next;
        }
    }
    return !base->is_stop_grad()
        && !base->flag(VarFlags::_requires_grad_disabled)
        && jittor::is_backward_leaf(base);
}

void VarHolder::orphan_views() {
    for (auto* w = views; w; ) {
        auto* next = w->next;
        // The record belongs to the holder that is the view, not to us; all we
        // may do is tell it that its base is gone.
        w->physical_storage = w->physical_storage
            && shares_physical_storage(w->owner->var, var);
        w->base = nullptr;
        w->prev = w->next = nullptr;
        w = next;
    }
    views = nullptr;
}

extern bool no_grad;
VarHolder* VarHolder::set_view_of(VarHolder* base, VarSlices&& slices,
                                bool inherit_requires_grad) {
    drop_view();
    if (!base) return this;
    for (int i=0; i<slices.n; i++)
        // An advanced index gathers, so its result is a copy -- and the Var*
        // it indexes with would outlive nothing in particular.
        if (slices.slices[i].is_var()) return this;
    if (inherit_requires_grad) {
        bool enabled = base->get_requires_grad();
        // A view whose creation recorded no edge can inherit a later source
        // activation. Its public flag must not create an accumulator or turn
        // an already constructed consumer into a recorded edge.
        bool unrecorded = no_grad || !enabled;
        if (unrecorded && var->is_stop_grad()
                && (var->dtype().is_float() || var->dtype().is_complex()))
            start_grad();
        set_requires_grad_impl(enabled);
        if (unrecorded)
            var->set_flag(VarFlags::_unrecorded_view);
    }
    attach_view(base, VarViewStep(move(slices)));
    return this;
}

void VarHolder::attach_view(VarHolder* base, VarViewStep final_step) {
    // Flatten: a view of a view is recorded against the root, so that the
    // intermediates of `y[1][2]` are free to die with the expression.
    bool physical = shares_physical_storage(var, base->var)
        && (!base->view || base->view->physical_storage);
    VarHolder* root = base;
    vector<VarViewStep> steps;
    if (base->view && base->view->base) {
        root = base->view->base;
        steps.reserve(base->view->steps.size() + 1);
        for (auto& step : base->view->steps) steps.push_back(step);
    }
    if (root == this) return;
    if (autograd_policy.stop_outputs_when_inputs_stopped) {
        auto source = base->ensure_autograd_alias();
        auto state = std::make_shared<AutogradAliasState>();
        state->source = source;
        state->version = source->version;
        state->creation_version = *state->version;
        state->created_without_grad = no_grad;
        state->recorded_view_history = !no_grad && backward_grad_fn(var);
        state->source_edge_enabled = base->get_requires_grad();
        state->inherit_requires_grad = no_grad || !base->get_requires_grad();
        state->own_requires_grad = !state->inherit_requires_grad && get_requires_grad();
        var->autograd_alias = move(state);
        if (var->autograd_alias->inherit_requires_grad) {
            if (var->is_stop_grad()
                    && (var->dtype().is_float() || var->dtype().is_complex())) {
                auto state = var->autograd_alias;
                bool own_requires_grad = state->own_requires_grad;
                start_grad();
                state->own_requires_grad = own_requires_grad;
                var->autograd_alias = move(state);
            }
            var->set_flag(VarFlags::_unrecorded_view);
            var->refresh_alias_requires_grad();
        }
    }
    steps.push_back(move(final_step));
    view = new VarView{root, this, move(steps), nullptr, root->views, physical};
    if (root->views) root->views->prev = view;
    root->views = view;
    retain_python_view_base();
}

VarHolder* VarHolder::set_transpose_view_of(VarHolder* base, NanoVector axes) {
    USER_CHECK(base) << "transpose view requires a source";
    auto rank = base->var->shape.size();
    USER_CHECK(axes.size() == rank) << "transpose view axes must match source rank";
    vector<bool> seen(rank, false);
    NanoVector normalized;
    for (int i=0; i<rank; ++i) {
        auto axis = axes[i] < 0 ? axes[i] + rank : axes[i];
        USER_CHECK(axis >= 0 && axis < rank && !seen[axis])
            << "transpose view axes must be a permutation";
        seen[axis] = true;
        normalized.push_back(axis);
    }
    drop_view();
    attach_view(base, VarViewStep(move(normalized)));
    return this;
}

bool VarHolder::is_last2_transpose_view() {
    if (!is_view() || view->steps.empty()) return false;
    const auto& step = view->steps.back();
    if (step.kind != VarViewStep::Transpose || step.axes.size() < 2) return false;
    int rank = step.axes.size();
    for (int i=0; i<rank; ++i) {
        int expected = i < rank-2 ? i : (i == rank-2 ? rank-1 : rank-2);
        if (step.axes[i] != expected) return false;
    }
    return true;
}

VarHolder* VarHolder::set_storage_view_of(VarHolder* base, bool expand) {
    USER_CHECK(base) << "Storage view requires a source";
    drop_view();
    attach_view(base, VarViewStep(expand ? VarViewStep::Expand : VarViewStep::Reshape,
                                  NanoVector(var->shape)));
    return this;
}

VarHolder* VarHolder::storage_permute(NanoVector axes) {
    return new VarHolder(storage_view_transpose(grad_input(), axes));
}

VarHolder* VarHolder::transpose_view_base() {
    USER_CHECK(is_last2_transpose_view()) << "tensor is not a live last-two-axis transpose view";
    VarPtr value(view->base->var);
    // Give the Python result its own Var/holder identity. The full basic view
    // aliases storage without stealing the source Var's holder back-pointer.
    if (view->steps.size() == 1)
        value = make_getitem(value.ptr, VarSlices(0));
    for (size_t i=0; i+1<view->steps.size(); ++i)
        value = apply_view_step(value.ptr, view->steps[i]);
    return new VarHolder(move(value));
}

NanoVector VarHolder::transpose_view_axes() {
    if (!is_view() || view->steps.empty()) return NanoVector();
    const auto& step = view->steps.back();
    if (step.kind != VarViewStep::Transpose) return NanoVector();
    return step.axes;
}

NanoVector VarHolder::producer_transpose_axes() {
    if (var->is_finished()) return NanoVector();
    Op* op = var->input();
    if (!op) return NanoVector();
    if (auto* transpose = dynamic_cast<TransposeOp*>(op)) return transpose->axes;
    if (auto* transpose = dynamic_cast<FuseTransposeOp*>(op)) return transpose->axes;
    return NanoVector();
}

VarHolder* VarHolder::transpose_view_source() {
    USER_CHECK(transpose_view_axes().size()) << "tensor is not a live transpose view";
    VarHolder* root = view->base;
    VarPtr value(root->var);
    if (view->steps.size() == 1)
        value = make_getitem(value.ptr, VarSlices(0));
    for (size_t i=0; i+1<view->steps.size(); ++i)
        value = apply_view_step(value.ptr, view->steps[i]);
    auto* result = new VarHolder(move(value));
    vector<VarViewStep> steps(view->steps.begin(), view->steps.end() - 1);
    // A view needs a step to write back through; the root itself is one
    // whole-shape reshape away.
    if (steps.empty())
        steps.emplace_back(VarViewStep::Reshape, NanoVector(root->var->shape));
    bool physical = shares_physical_storage(result->var, root->var)
        && (!root->view || root->view->physical_storage);
    result->view = new VarView{root, result, move(steps), nullptr, root->views, physical};
    if (root->views) root->views->prev = result->view;
    root->views = result->view;
    return result;
}

// Whether one recorded view step still applies to `value`.
//
// A view is a lazy expression over its base's data, so it can only be
// re-derived while that data has the shape the steps were recorded against.
// A rebind can change the number of elements -- `x.data = y` allows it, and
// vLLM-Omni's layerwise offload swaps a parameter for a zero-element
// placeholder -- and re-applying a reshape or expand step to the new data then
// aborts the process (reshape_op.cc: "reshape shape is invalid for input of
// size [x_items(0) == y_items(1152)]"). Slices are left alone: they were
// already only ever recorded against a compatible shape.
static bool view_step_fits(Var* value, const VarViewStep& step) {
    if (step.kind == VarViewStep::Slice) return true;
    // A transpose step stores the axes permutation, not a shape: it applies to
    // any value of the same rank.
    if (step.kind == VarViewStep::Transpose)
        return (int)step.axes.size() == value->shape.size();
    // Reshape and Expand both record the *target* shape in `axes`, and each has
    // to be tested against what it actually constrains. Neither constrains the
    // base's rank, and only one constrains the element count.
    //
    // Testing both against both is what this did before, and it rejected the
    // ordinary cases rather than the broken ones: a reshape's target rank is
    // unrelated to its base's by definition, so `x.view(-1)` on anything of
    // rank > 1 failed the rank comparison; and an expand's target holds MORE
    // elements than its source by construction, so every widening expand failed
    // the element-count comparison. A rejected step drops the view, which
    // freezes it at whatever data it last saw and silently stops write-through
    // from ever reaching the base again -- no error, no warning.
    if (step.kind == VarViewStep::Reshape) {
        int64 target = 1;
        for (int i = 0; i < (int)step.axes.size(); i++) target *= step.axes[i];
        return target == value->num;
    }
    // Expand: the target is still usable when, aligned from the right, every
    // source axis either already matches it or is 1 and can be widened to it.
    // Extra leading target axes are exactly what broadcasting introduces.
    int extra = (int)step.axes.size() - (int)value->shape.size();
    if (extra < 0) return false;
    for (int i = 0; i < (int)value->shape.size(); i++)
        if (value->shape[i] != step.axes[i + extra] && value->shape[i] != 1)
            return false;
    return true;
}

void VarHolder::refresh_transpose_views(bool preserve_grad_policy, VarView* only) {
    for (auto* record = views; record; ) {
        // drop_view() below unlinks and frees `record`, so step off it first.
        auto* next = record->next;
        if (only && record != only) {
            record = next;
            continue;
        }
        auto state = record->owner->var->autograd_alias;
        bool rebase = state && !state->created_without_grad && state->source
            && (state->recorded_view_history || state->source->requires_grad())
            && state->creation_version != *state->version;
        // Rebuilding a view after an unrecorded store is graph bookkeeping,
        // not a new user operation under that store's no-grad scope. Views
        // originally created with recording enabled keep a recorded producer.
        struct RecordingScope {
            bool previous;
            Var* source;
            bool disabled;
            RecordingScope(bool record, Var* source, bool freeze_edge)
                : previous(no_grad), source(source),
                  disabled(source->flag(VarFlags::_requires_grad_disabled)) {
                if (record) no_grad = false;
                if (freeze_edge) source->set_flag(VarFlags::_requires_grad_disabled);
            }
            ~RecordingScope() {
                source->set_flag(VarFlags::_requires_grad_disabled, disabled);
                no_grad = previous;
            }
        } recording(rebase, var, rebase && state->recorded_view_history
                    && !get_requires_grad());
        VarPtr value(var);
        bool fits = true;
        bool physical = !view || view->physical_storage;
        for (const auto& step : record->steps) {
            if (!view_step_fits(value.ptr, step)) {
                fits = false;
                break;
            }
            VarPtr next_value = apply_view_step(value.ptr, step,
                                                record->physical_storage);
            physical = physical && shares_physical_storage(next_value.ptr, value.ptr);
            value = move(next_value);
        }
        if (fits) {
            if (state && !rebase
                    && record->owner->var->flag(VarFlags::_unrecorded_view))
                value = jittor::detach(value.ptr);
            record->owner->rebind_view_result(move(value), preserve_grad_policy);
            if (rebase) {
                state->source_edge_enabled = get_requires_grad();
                state->recorded_view_history = true;
                state->inherit_requires_grad = false;
                state->own_requires_grad = true;
                auto* current = record->owner->var;
                current->set_flag(VarFlags::_unrecorded_view, 0);
                current->set_flag(VarFlags::_requires_grad_disabled, 0);
            }
            record->physical_storage = physical;
        } else if (record->owner) {
            // The view keeps the data it was taken from, which is what torch's
            // views do after `x.data = y` replaces the storage under them.
            record->owner->drop_view();
        }
        record = next;
    }
}

static VarPtr view_assignment_value(Var* base, const vector<VarViewStep>& steps, Var* value) {
    const int n = steps.size();
    // What each step is applied to. The first is the base itself; the rest are
    // the intermediates, rebuilt rather than remembered.
    vector<VarPtr> targets(n);
    Var* cur = base;
    for (int i=0; i<n-1; i++) {
        targets[i] = apply_view_step(cur, steps[i]);
        cur = targets[i].ptr;
    }
    // Fold the write back outwards: the innermost slice takes `value`, and each
    // level's result is what the level above writes into its own slice.
    VarPtr updated;
    for (int i=n-1; i>=0; i--) {
        Var* target = i ? targets[i-1].ptr : base;
        if (steps[i].kind == VarViewStep::Transpose) {
            const auto& axes = steps[i].axes;
            vector<int64> inverse_values(axes.size());
            for (int j=0; j<axes.size(); ++j) inverse_values[axes[j]] = j;
            NanoVector inverse = NanoVector::make(inverse_values.data(), inverse_values.size());
            updated = make_transpose_view(value, move(inverse));
        } else if (steps[i].kind == VarViewStep::Expand || steps[i].kind == VarViewStep::Reshape) {
            USER_CHECK(value->num == target->num)
                << "Cannot write to an expanded view with overlapping storage";
            updated = make_storage_reshape(value, NanoVector(target->shape));
        } else {
            updated = make_setitem(target, VarSlices(steps[i].slices), value, ns_void);
        }
        value = updated.ptr;
    }
    return updated;
}

bool VarHolder::write_through_view(Var* value, bool preserve_grad_policy) {
    if (!view || !view->base) return false;
    auto updated = view_assignment_value(view->base->var, view->steps, value);
    view->base->rebind_view_result(move(updated), preserve_grad_policy);
    return true;
}

bool VarHolder::copy_view_into(VarHolder* src) {
    check_inplace_autograd();
    ExecutorEntryScope entry;
    auto* base = view && view->base ? view->base : this;
    bool materialized = false;
    for (auto* record=base->views; record; record=record->next)
        materialized = materialized || !record->owner->storage_view_is_physical();
    if (!materialized) return false;
    // Backend copies may release the GIL. Pin the original Python holders
    // and copy the weak view records before any synchronization can do so.
    PyObjHolder owners(PyList_New(0));
    CHECKop(PyList_Append(owners.obj, GET_OBJ_FROM_RAW_PTR(base)), ==, 0);
    vector<std::pair<VarHolder*, vector<VarViewStep>>> records;
    for (auto* record=base->views; record; record=record->next) {
        CHECKop(PyList_Append(owners.obj, GET_OBJ_FROM_RAW_PTR(record->owner)), ==, 0);
        if (!record->owner->storage_view_is_physical())
            records.emplace_back(record->owner, record->steps);
    }
    vector<VarViewStep> target_steps;
    if (view && view->base) target_steps = view->steps;
    // The caller has drained pending readers. Keep every existing holder and
    // graph node, preparing the complete write on independent device storage.
    sync(false, false);
    src->sync(false, false);
    validate_logical_copy_alias(this, src);
    if (validate_storage_copy(var, src->var) == StorageCopy::NoOp) {
        record_inplace_write();
        return true;
    }
    static auto make_copy = op_constructor<VarPtr, Var*>("copy");
    VarPtr staged;
    if (!target_steps.empty()) {
        auto snapshot = make_copy(base->var);
        staged = view_assignment_value(snapshot.ptr, target_steps, src->var);
    } else {
        staged = make_copy(src->var);
    }
    unique_ptr<VarHolder> updated(new VarHolder(move(staged)));
    updated->sync(false, false);
    validate_storage_copy(base->var, updated->var);
    vector<VarHolder*> destinations;
    vector<unique_ptr<VarHolder>> unique_destinations;
    vector<unique_ptr<VarHolder>> values;
    for (auto& record : records) {
        VarPtr value(updated->var);
        for (const auto& step : record.second) {
            USER_CHECK(view_step_fits(value.ptr, step))
                << "_copy_view_into cannot refresh an incompatible retained view";
            value = apply_view_step(value.ptr, step);
        }
        unique_ptr<VarHolder> prepared(new VarHolder(move(value)));
        prepared->sync(false, false);
        auto* destination = record.first;
        // A read-only expanded sibling repeats the same backing elements;
        // refreshing that backing storage is valid even though writing an
        // arbitrary expanded destination above is refused. Select each
        // repeated axis once, through the existing physical getitem owner.
        VarSlices unique(destination->var->shape.size());
        bool repeated = false;
        for (int d=0; d<unique.n; ++d) {
            if (destination->var->shape[d] > 1 && !destination->var->storage_stride(d)) {
                unique.slices[d].set_int(0);
                repeated = true;
            } else {
                unique.slices[d].slice = Slice{0, 0, 1, 3};
            }
        }
        if (repeated) {
            unique_ptr<VarHolder> backing(new VarHolder(
                make_getitem(destination->var, VarSlices(unique))));
            backing->sync(false, false);
            USER_CHECK(shares_physical_storage(backing->var, destination->var))
                << "_copy_view_into needs a physical expanded backing view";
            prepared.reset(new VarHolder(make_getitem(prepared->var, move(unique))));
            prepared->sync(false, false);
            destination = backing.get();
            unique_destinations.push_back(move(backing));
        }
        // Validate all destinations before changing the first original byte.
        validate_storage_copy(destination->var, prepared->var);
        destinations.push_back(destination);
        values.push_back(move(prepared));
    }
    base->copy_into(updated.get());
    for (size_t i=0; i<destinations.size(); ++i)
        destinations[i]->copy_into(values[i].get());
    return true;
}

VarHolder::~VarHolder() {
    drop_view();
    orphan_views();
    if (PREDICT_BRANCH_NOT_TAKEN(!var)) return;
    if (PREDICT_BRANCH_NOT_TAKEN(state_capture != nullptr))
        state_capture->erase(this);
    drop_capture_record(this);
    unlink_from_hold_vars(iter);
    release_holder();
    // Dropping the last holder runs the liveness propagation, which frees
    // nodes, which reaches the allocator: every one of those steps reports by
    // throwing. A destructor is implicitly noexcept, so an escaping error is
    // std::terminate *at this frame* -- the generated tp_dealloc wraps the
    // call in a try, but that catch is downstream of the terminate and never
    // runs. Report and carry on, which is the teardown rule for the rest of
    // the tree (CHECK_ACL_PEEK, peekCudaErrorsAlways).
    try {
        var->release_both_liveness();
    } catch (const std::exception& e) {
        LOGe << "error while releasing a Var, ignored during teardown:"
            << e.what();
    }
}

// assign attributes of b to a
static inline void assign_var(Var* a, Var* b, bool preserve_grad_policy = true) {
    a->autograd_alias = b->autograd_alias;
    if (!preserve_grad_policy && a->autograd_alias) {
        // A real RHS edge can activate a formerly frozen destination. Keep
        // the shared write version, but publish the replacement's actual
        // policy before rebuilding its sibling views.
        a->autograd_alias->own_requires_grad = !a->is_stop_grad()
            && !a->flag(VarFlags::_requires_grad_disabled);
        a->autograd_alias->inherit_requires_grad = false;
    }
    if (b->grad_hook) a->grad_hook = b->grad_hook;
    if (b->flag(VarFlags::_placement_published))
        a->set_flag(VarFlags::_placement_published);
    a->name = move(b->name);
    if (preserve_grad_policy && b->is_stop_grad())
        a->set_stop_grad();
    if (b->flag(VarFlags::_stop_fuse))
        a->set_flag(VarFlags::_stop_fuse);
    if (preserve_grad_policy && b->flag(VarFlags::_explicit_requires_grad))
        a->set_flag(VarFlags::_explicit_requires_grad);
    if (preserve_grad_policy)
        a->set_flag(VarFlags::_requires_grad_disabled,
            b->flag(VarFlags::_requires_grad_disabled));
    if (preserve_grad_policy)
        a->set_flag(VarFlags::_unrecorded_view,
            b->flag(VarFlags::_unrecorded_view));
}

void VarHolder::operator=(VarPtr&& v) {
    rebind_view_result(move(v), true);
}

void VarHolder::rebind_view_result(VarPtr&& v, bool preserve_grad_policy) {
    if (preserve_grad_policy && autograd_policy.preserve_requires_grad_on_assignment) {
        if (var->is_stop_grad() != v->is_stop_grad())
            v.set_stop_grad(var->is_stop_grad());
        if (var->flag(VarFlags::_explicit_requires_grad))
            v.ptr->set_flag(VarFlags::_explicit_requires_grad);
    }
    note_rebind(this);
    drop_capture_record(this);
    assign_var(v.ptr, var, preserve_grad_policy);
    release_holder();
    var->release_both_liveness();
    var = v.ptr;
    own_holder();
    v.ptr = nullptr;
    refresh_transpose_views(preserve_grad_policy);
}

extern bool no_grad;
void VarHolder::set_requires_grad(bool flag) {
    if (autograd_policy.stop_outputs_when_inputs_stopped
            && !flag && get_requires_grad()) {
        var->check_alias_history();
        USER_CHECK(jittor::is_backward_leaf(var))
            << "you can only change requires_grad flags of leaf variables.";
    }
    set_requires_grad_impl(flag);
}

void VarHolder::set_requires_grad_impl(bool flag) {
    auto state = var->autograd_alias;
    if (state) {
        state->own_requires_grad = flag;
    }
    // Explicit activation gives an unrecorded view its own gradient edge.
    // This must precede the equal-value fast path: requires_grad_(True) is
    // meaningful even when the inherited public flag is already true.
    if (flag) var->set_flag(VarFlags::_unrecorded_view, 0);
    // Turning off an inherited flag does not disable the source and does not
    // give this otherwise unrecorded view an accumulator.
    if (state && state->inherit_requires_grad) {
        var->set_flag(VarFlags::_unrecorded_view, !flag);
        var->refresh_alias_requires_grad();
        return;
    }
    if (flag == get_requires_grad()) return;
    if (flag) {
        if (var->is_stop_grad()) {
            start_grad();
        } else {
            // Keep the same Var node so graphs built before a temporary freeze
            // become differentiable again when the flag is restored.
            var->set_flag(VarFlags::_requires_grad_disabled, 0);
        }
    } else {
        // stop_grad() releases backward liveness and is intentionally permanent.
        // requires_grad_(False) is a reversible leaf policy: existing graph edges
        // stay alive, while newly initialized Ops snapshot disabled input edges.
        var->set_flag(VarFlags::_requires_grad_disabled);
    }
}

std::shared_ptr<AutogradAliasState> VarHolder::ensure_autograd_alias() {
    if (!var->autograd_alias) {
        var->autograd_alias = std::make_shared<AutogradAliasState>();
        var->autograd_alias->own_requires_grad = get_requires_grad();
    }
    return var->autograd_alias;
}

VarHolder* VarHolder::check_inplace_autograd() {
    if (no_grad || !autograd_policy.stop_outputs_when_inputs_stopped) return this;
    if (get_requires_grad() && var->autograd_alias
            && var->autograd_alias->created_without_grad
            && var->autograd_alias->source
            && var->autograd_alias->source->requires_grad()) {
        USER_CHECK(false) << "A view was created in no_grad mode and is being "
            "modified inplace with grad mode enabled.";
    }
    if (get_requires_grad()) {
        USER_CHECK(!jittor::is_backward_leaf(var))
            << "a leaf Variable that requires grad is being used in an in-place operation.";
        USER_CHECK(!view_base_is_trainable_leaf())
            << "a view of a leaf Variable that requires grad is being used in an in-place operation.";
    }
    return this;
}

void VarHolder::record_inplace_write() {
    if (!var->autograd_alias) return;
    ++*var->autograd_alias->version;

}

// Physical view writeback can run eagerly, but consuming autograd history is
// observable: Torch rebases a view using the source policy at its first real
// history query or grad-mode use, not at requires_grad_ or storage inspection.
void VarHolder::observe_alias_history() {
    var->refresh_alias_requires_grad();
    var->check_alias_history();
    auto state = var->autograd_alias;
    if (!state || state->created_without_grad || !state->source
            || state->creation_version == *state->version) return;
    // A view that has never recorded a producer still has nothing to query
    // while its source is frozen. Preserve the pending epoch: activating the
    // source later can then rebase this actual write, unlike an untouched view.
    if (!state->recorded_view_history && !state->source->requires_grad()) return;
    if (view && view->base && view->physical_storage
            && state->recorded_view_history
            && state->source_edge_enabled == view->base->get_requires_grad()
            && shares_physical_storage(var, view->base->var)) {
        // An imperative store changed bytes and the shared version, not this
        // physical view's allocation or recorded source policy. Keep its real
        // producer and materialized buffer. Rebuilding it here would discard
        // the original AD owner merely because somebody inspected is_leaf.
        state->creation_version = *state->version;
        return;
    }
    if (view && view->base
            && (state->recorded_view_history || state->source->requires_grad()))
        view->base->refresh_transpose_views(true, view);
    state->creation_version = *state->version;
}

Var* VarHolder::grad_input() {
    if (!no_grad) observe_alias_history();
    return var;
}

VarHolder* VarHolder::storage_data_alias() {
    auto* alias = new VarHolder(jittor::detach(var));
    alias->stop_grad();
    // Tensor.data shares bytes but has an independent version counter and
    // trainability. Detach above shares the counter for its distinct contract.
    alias->var->autograd_alias = std::make_shared<AutogradAliasState>();
    return alias;
}

bool VarHolder::has_leaf_accumulator() {
    if (!get_requires_grad() || var->flag(VarFlags::_unrecorded_view)) return false;
    auto state = var->autograd_alias;
    if (state && (state->stale_unrecorded_view()
            || (state->recorded_view_history
                && state->creation_version != *state->version))) return false;
    return jittor::is_backward_leaf(var);
}

bool VarHolder::producer_is_view() {
    if (var->is_finished()) return true;
    Op* op = var->input();
    return op && op->is_storage_view();
}

string VarHolder::producer_unary() {
    if (var->is_finished()) return "";
    Op* op = var->input();
    if (!op || !op->is_op(op_ids::unary()) || op->inputs().size() != 1) return "";
    if (op->inputs().front()->shape != var->shape) return "";
    if (op->ns == ns_cast) return var->dtype().to_cstring();
    return op->ns.to_cstring();
}

string VarHolder::producer_name() {
    if (var->is_finished()) return "";
    Op* op = var->input();
    return op ? string(op->name()) : "";
}

string VarHolder::producer_op() {
    if (var->is_finished()) return "";
    Op* op = var->input();
    return op ? op->name_ex() : "";
}

bool VarHolder::is_pending_contiguous() {
    if (var->is_finished()) return false;
    Op* op = var->input();
    return op && op->is_op(op_ids::contiguous()) && op->inputs().size() == 1;
}

void mark_python_number(VarHolder* holder) {
    holder->var->set_flag(VarFlags::_python_number);
}

VarHolder* VarHolder::start_grad() {
    if (!var->dtype().is_float() && !var->dtype().is_complex()) {
        std::stringstream message;
        message << "requires_grad has no effect on this " << var->dtype() << var->shape
            << " Var: only floating-point and complex values have gradients.";
        user_warning(message.str());
    }
    bool no_grad_bk = no_grad;
    AutogradPolicyOverride policy_guard({});
    no_grad = 0;
    auto alias_state = var->autograd_alias;
    auto dvar = jittor::detach(var);
    dvar->grad_hook = var->grad_hook;
    std::swap(dvar.ptr, var);
    var->autograd_alias = move(alias_state);
    if (var->autograd_alias) {
        var->autograd_alias->own_requires_grad = true;
    }
    no_grad = no_grad_bk;
    var->set_flag(VarFlags::_explicit_requires_grad);
    var->set_flag(VarFlags::_requires_grad_disabled, 0);
    return this;
}

bool VarHolder::is_backward_leaf() {
    observe_alias_history();
    return jittor::is_backward_leaf(var);
}

void VarHolder::set_grad_hook(GradCallback&& callback) {
    var->grad_hook = std::make_shared<GradCallback>(move(callback));
}

void VarHolder::clear_grad_hook() {
    var->grad_hook.reset();
}

PyObject* VarHolder::collectable_grad_hook() {
    if (!var->_outputs.empty() ||
        !var->grad_hook || var->grad_hook.use_count() != 1)
        return nullptr;
    // `Var::holder` is the most recently constructed holder and temporary
    // aliases clear it on destruction. Check the actual ownership registry.
    for (auto holder : runtime_holder_state().holders())
        if (holder != this && holder->var == var) return nullptr;
    return reinterpret_cast<PyObject*>(var->grad_hook->python_callable);
}

void VarHolder::clear_collectable_grad_hook() {
    if (collectable_grad_hook()) var->grad_hook.reset();
}

int64 VarHolder::grad_fn_node_id() {
    observe_alias_history();
    Op* op = backward_grad_fn(var);
    return op ? op->id : -1;
}

int64 VarHolder::grad_fn_op_id() {
    observe_alias_history();
    Op* op = backward_grad_fn(var);
    if (!op) return -1;
    // Op::type_id() resolves through get_op_info, which raises for a name that
    // was never registered. A query behind an attribute read must not raise, so
    // an unregistered op answers with the id OpInfo reserves for "unresolved".
    if (!op->registered_op_id && !has_op(op->name())) return 0;
    return op->type_id();
}

string VarHolder::grad_fn_name() {
    observe_alias_history();
    Op* op = backward_grad_fn(var);
    return op ? op->name_ex() : string();
}

string VarHolder::to_string() {
    return var->to_string();
}

VarHolder* VarHolder::assign(VarHolder* v) {
    return assign_impl(v, true);
}

VarHolder* VarHolder::assign_impl(VarHolder* v, bool preserve_grad_policy) {
    if (preserve_grad_policy && autograd_policy.preserve_requires_grad_on_assignment) {
        v->set_requires_grad_impl(get_requires_grad());
    }
    VarPtr replacement(v->var);
    if (preserve_grad_policy && autograd_policy.preserve_requires_grad_on_assignment
            && !var->is_stop_grad() && replacement->is_stop_grad()) {
        // A reversibly frozen destination must not become permanently stopped
        // because its replacement was computed under no-grad. Use the same
        // native storage alias normalization as rebind_view_result; the old
        // disabled flag below still snapshots the frozen input edge.
        replacement.set_stop_grad(false);
    }
    record_inplace_write();
    // `assign` is the in-place primitive every `x.foo_()` funnels through, so
    // this is the one place that has to know that an in-place write to a view
    // is a write to the thing it is a view of.
    // Rebinding the base already refreshes this holder from the updated base.
    // Keep that graph: replacing it again by the bare RHS would drop the old
    // base's zero-gradient edge for a slice copy (Torch's CopySlices rule).
    if (write_through_view(replacement.ptr, preserve_grad_policy)) return this;
    note_rebind(this);
    drop_capture_record(this);
    assign_var(replacement.ptr, var, preserve_grad_policy);
    release_holder();
    replacement->own_both_liveness();
    var->release_both_liveness();
    var = replacement.ptr;
    own_holder();
    refresh_transpose_views(preserve_grad_policy);
    return this;
}

VarHolder* VarHolder::update(VarHolder* v) {
    v->var->set_flag(VarFlags::_out_hint);
    return assign(v);
}

VarHolder* VarHolder::_update(VarHolder* v, bool logical_write) {
    if (!logical_write) {
        ExecutorEntryScope entry;
        // Finish readers of the old storage before replacing it. The Var and
        // its producer remain the same: recorded consumers and saved backward
        // inputs still refer to this tensor, including on a nonleaf tensor.
        // A detached storage owner also pins the RHS for self/overlap cases.
        auto replacement = jittor::detach(v->var);
        sync_all(false);
        runtime_executor().run_sync({var, replacement.ptr}, false, false);
        // Existing views retain their old allocation. Retain this tensor's
        // view provenance for autograd guards, but detach its writeback owner.
        if (view) unlink_view_owner(view);
        orphan_views();
        if (var->shape != replacement->shape && !var->recorded_gradient_shape) {
            bool recorded = backward_grad_fn(var) != nullptr;
            for (Op* consumer : var->outputs())
                recorded = recorded || (!consumer->is_stop_grad()
                    && !is_requires_grad_disabled_edge(var, consumer));
            if (recorded)
                var->recorded_gradient_shape = std::make_shared<const NanoVector>(var->shape);
        }
        if (var->allocator) free_var_mem(var);
        var->dtype() = replacement->dtype();
        var->set_shape(replacement->shape);
        var->storage_strides = replacement->storage_strides;
        var->device_id = replacement->device_id;
        var->placement = replacement->placement;
        var->share_with(replacement.ptr);
        CHECK(var->alloc(replacement->allocator));
        return this;
    }
    // Python in-place wrappers call `_update` directly. A view must write the
    // replacement through its root instead of rebinding only the view holder.
    if (is_view()) {
        // A previously frozen destination can acquire an RHS autograd edge.
        // Preserve that edge through both the base rebind and its view refresh
        // instead of copying the destination's old stopped policy onto it.
        if (autograd_policy.preserve_requires_grad_on_assignment
                && !get_requires_grad() && v->get_requires_grad()) {
            return assign_impl(v, false);
        }
        return assign(v);
    }
    check_inplace_autograd();
    record_inplace_write();
    v->var->autograd_alias = var->autograd_alias;
    if (var->grad_hook) v->var->grad_hook = var->grad_hook;
    if (var->flag(VarFlags::_placement_published))
        v->var->set_flag(VarFlags::_placement_published);
    note_rebind(this);
    drop_capture_record(this);
    release_holder();
    v->var->own_both_liveness();
    var->release_both_liveness();
    var = v->var;
    own_holder();
    var->set_flag(VarFlags::_out_hint);
    refresh_transpose_views();
    return this;
}


VarHolder* VarHolder::sync(bool device_sync, bool weak_sync) {
    jittor::sync({this}, device_sync, weak_sync);
    return this;
}

ArrayArgs VarHolder::fetch_sync() {
    note_readback();
    if (!host_value_ready(var)) {
        ExecutorEntryScope entry;
        sync(readback_waits_for_devices());
        var->set_flag(VarFlags::_execution_failed, 0);
        if (save_mem || _HAS_ACCELERATOR)
            migrate_to_cpu(var, runtime_executor().allocator);
    }
    ArrayArgs result{var->mem_ptr, var->shape, var->dtype()};
    result.storage_strides = var->storage_strides;
    return result;
}

inline static void cast_item_data(ItemData& data) {
    // The conversion and the dtype rewrite must stay in the same branch: the
    // bfloat16 arm used to be compiled out on ROCm while `dtype = ns_float32`
    // sat outside the #ifndef, so a ROCm bf16 scalar was handed to Python as
    // the raw bit pattern reinterpreted as float32.
    if (data.dtype == ns_float16) {
        auto* fp16 = (float16*)&data;
        float32 value = float32(fp16[0]);
        auto* fp32 = (float32*)&data;
        fp32[0] = value;
        data.dtype = ns_float32;
    } else if (data.dtype == ns_bfloat16) {
        #ifndef IS_ROCM
        auto* bf16 = (bfloat16*)&data;
        float32 value = float32(bf16[0]);
        #else
        // ROCm has no host-side bfloat16 -> float32 conversion operator, but
        // bfloat16 is the high half of a float32: widening is an exact
        // 16-bit shift of the bit pattern.
        uint32 bits = uint32(*(uint16*)&data) << 16;
        float32 value;
        std::memcpy(&value, &bits, sizeof(value));
        #endif
        auto* fp32 = (float32*)&data;
        fp32[0] = value;
        data.dtype = ns_float32;
    }
}

ItemData VarHolder::item() {
    note_readback();
    // Keep the entry lock through the final scalar copy, including managed
    // allocations which migrate_to_cpu deliberately leaves on the device.
    ExecutorEntryScope entry;
    USER_CHECK(var->num==1) << "item() needs a Var with exactly one element, but this one has"
        << var->num << "(shape" << var->shape >> ")";
    // Value-initialize: only dsize bytes are written below, and the converter
    // may read all 8 (unsigned dtypes go through PyLong_FromUnsignedLongLong).
    ItemData data{};
    data.dtype = var->dtype();
    auto dsize = data.dtype.dsize();
    if (!host_value_ready(var)) {
        // A blocking backend host copy waits for its producer stream.
        sync(false, false);
        var->set_flag(VarFlags::_execution_failed, 0);
        // An element that shares its storage is read where it lies, below.
        // Moving it to the host moves its whole storage group -- for a view
        // `x[0:1]`, all of `x` -- and the next kernel to read `x` moved it
        // back, with a blocking copy either way: 18.9 us an `x[0:1].item()`
        // against 7.7 on 1.3, whose managed memory made the move a no-op.
        // A var of its own still moves: that is the element alone, and every
        // later read of it is then a host copy rather than a readback.
        bool in_place = var->share_next && var->mem_ptr && var->allocator->is_cuda();
        if (save_mem || (_HAS_ACCELERATOR && !in_place))
            migrate_to_cpu(var, runtime_executor().allocator);
    }
    #ifdef HAS_ACCELERATOR
    if (var->allocator->is_cuda()) {
        backend_copy(&data.data, {BackendId::Cpu, 0}, var->mem_ptr,
                     allocation_device(var->allocator), dsize);
    } else
    #endif
    {
        std::memcpy(&data.data, var->mem_ptr, dsize);
    }
    if (data.dtype == ns_float16 || data.dtype == ns_bfloat16)
        cast_item_data(data);
    return data;
}

// from fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher;

void sync_all(bool device_sync) {
    vector<Var*> vars;
    vars.reserve(runtime_holder_state().holders().size());
    for (auto v : runtime_holder_state().holders()) {
        // Same reason `top_weak_sync` skips these: a kept graph is run on
        // purpose by whoever kept it. `sync_all` cannot finish one anyway --
        // `keep_graph` is what leaves it pending -- so sweeping it up here
        // only re-runs it, every time anyone asks for everything to complete.
        if (v->var->flag(VarFlags::_kept)) continue;
        // Sinks only. Sweeping every holder and forcing weak_sync=false were
        // both tried against MiniMax-H3's uniform-random-byte video: ~70%
        // failure becomes ~17% and ~33% respectively, and ~17% with both. They
        // raise the chance the Var is evaluated in passing; they do not fix it,
        // so the root cause is not here. See section 48 of the H3 results doc.
        if (!v->var->_outputs.size())
            vars.push_back(v->var);
    }
    for (auto& v :fetcher)
        vars.push_back(v.ptr);
    graph_check();
    runtime_executor().run_sync(vars, device_sync); //need sync at last
    graph_check();
}

void sync(const vector<VarHolder*>& vh, bool device_sync, bool weak_sync) {
    vector<Var*> vars;
    vars.reserve(vh.size());
    for (auto v : vh) vars.push_back(v->var);
    graph_check();
    runtime_executor().run_sync(vars, device_sync, weak_sync); //need sync at last
    graph_check();
}

vector<ArrayArgs> fetch_sync(const vector<VarHolder*>& vh) {
    note_readback();
    vector<ArrayArgs> ret(vh.size());
    ExecutorEntryScope entry;
    sync(vh, readback_waits_for_devices());
    for (uint i=0; i<vh.size(); i++) {
        if (save_mem || _HAS_ACCELERATOR)
            migrate_to_cpu(vh[i]->var, runtime_executor().allocator);
        ret[i].ptr = vh[i]->var->mem_ptr;
        ret[i].shape = vh[i]->var->shape;
        ret[i].dtype = vh[i]->var->dtype();
        ret[i].storage_strides = vh[i]->var->storage_strides;
    }
    return ret;
}

string VarHolder::debug_msg() {
    std::stringstream ss;
    ss << var;
    return ss.str();
}

int VarHolder::grad() {
    USER_ERROR << R""(Jittor Var doesn't have this interface, please change
your code as below::

    model = Model()
    optimizer = SGD(model.parameters())
    ...
    optimizer.backward(loss)
    
    for p in model.parameters():
        # prev code:
        # grad = p.grad

        # change to:
        grad = p.opt_grad(optimizer)
)"";
    return 0;
}


static auto make_ternary = op_constructor<VarPtr, Var*, Var*, Var*>("ternary");

extern bool no_grad;

VarHolder* ternary_out_hint(VarHolder* cond, VarHolder* x, VarHolder* y) {
    if (!no_grad)
        cond->var->set_flag(VarFlags::_out_hint);
    return new VarHolder(make_ternary(cond->var, x->var, y->var));
}

void migrate_all_to_cpu() {
    sync_all(true);
    if (save_mem || _HAS_ACCELERATOR)
        for (auto vh : runtime_holder_state().holders()) {
            auto v = vh->var;
            // if (v->_outputs.size()) continue;
            if (v->allocator && v->mem_ptr && !v->allocator->is_cuda())
                migrate_to_cpu(v, cpu_allocator);
        }
}

} // jittor
