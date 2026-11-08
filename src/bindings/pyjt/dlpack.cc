// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// DLPack ownership/device/stream semantics.
//
//  - Export (Var -> DLManagedTensor capsule): run the Var's pending work and
//    honour the consumer's `__dlpack__(stream=)` (parse_export_stream():
//    None waits for the device, -1 does not synchronize, a stream handle is
//    made to wait for Jittor's compute stream through an event), read its
//    real device/dtype/shape/strides via the already-completed per-tensor
//    device id infrastructure (VarHolder::location()/device_id()), and stash
//    a VarPtr in manager_ctx so the underlying storage outlives the capsule
//    even if the Python VarHolder is immediately dropped.
//  - Import (capsule -> Var): reuse the exact "hand-built Var over foreign
//    memory" pattern already proven by reuse_np_array() (py_array_op.cc),
//    but route CUDA-sourced tensors through the per-device ForeignAllocator
//    (mem/allocator/foreign_allocator.h) instead of the plain CPU-only one,
//    so location()/device_id() report the truth instead of silently lying
//    that GPU memory is CPU memory. The imported Var is explicitly pinned to
//    its real device (an explicit TensorPlacement, mirroring device_copy_op)
//    so later ops route to the correct physical device without depending on
//    the ambient global flag.
//  - Ownership: PyCapsule's own single-owner semantics do the "consume
//    exactly once" enforcement (rename "dltensor"->"used_dltensor" on
//    import; a capsule destructor fires the same deleter exactly once if
//    the capsule is garbage-collected without ever being imported).
//  - Layout: a strided Var (a transpose, step slice or broadcast view of a
//    storage) exports its own element strides over the same bytes; copy=True
//    exports a dense independent buffer instead. On import, a contiguous or
//    non-negatively strided producer is aliased zero-copy, as a storage view
//    when strided. A negative stride, which a Jittor Var cannot express, is
//    refused here; jt.from_dlpack() then imports the flat element span via
//    from_dlpack_capsule(capsule, flat_len, elem_offset) and gathers the
//    logical tensor out of it (device-side, an independent copy).
//    dlpack_peek() reports shape, strides and aliasability without consuming
//    the capsule, so the Python wrapper chooses before committing.
//  - Versioning: both the classic (unversioned) DLManagedTensor and the
//    DLPack>=0.8 DLManagedTensorVersioned capsule are supported, on export
//    and import. Export only produces the versioned form when the consumer
//    explicitly opts in via __dlpack__(max_version=(major,minor)) with
//    major>=1; the default remains the classic capsule (DLPack's own
//    mandated fallback, and what NumPy/CuPy/PyTorch's own from_dlpack()
//    consume when they don't request a version). Import accepts either
//    capsule name a producer hands back and, for the versioned form, checks
//    DLPackVersion.major before touching any other field (a major mismatch
//    only guarantees the deleter is safe to call, per spec).
//  - copy=True: honored by materializing an independent buffer (a real,
//    backend-ordered copy, not an aliasing op) instead of rejecting the
//    request -- see make_materialized_copy().
//
// Unlike a from-scratch CUDA-only implementation, every device-touching step
// here goes through the generic runtime/backend.h API (backend_copy,
// backend_synchronize, accelerator_backend_id, backend_raw_allocator) rather
// than a direct CUDA call, so this works unmodified on whichever accelerator
// backend (CUDA, ROCm, ACL, ...) this build was compiled for.
#include <Python.h>
#include <vector>
#include <cstring>
#include <functional>
#include "bindings/pyjt/dlpack.h"
#include "bindings/pyjt/dlpack_api.h"
#include "core/var_holder.h"
#include "core/var.h"
#include "mem/allocator.h"
#include "mem/allocator/foreign_allocator.h"
#include "runtime/backend.h"
#include "runtime/tensor_placement.h"

namespace jittor {

// ---------------------------------------------------------------------
// dtype <-> DLDataType
// ---------------------------------------------------------------------

static DLDataType ns_to_dltype(NanoString ns) {
    DLDataType dt;
    dt.lanes = 1;
    dt.bits = (uint8_t)(ns.dsize() * 8);
    // is_complex()/is_float() both read plain bit flags that don't
    // distinguish bfloat16 from the other floats, nor complex from real --
    // order matters here.
    if (ns.is_complex()) dt.code = kDLComplex;
    else if (ns == ns_bfloat16) dt.code = kDLBfloat;
    else if (ns.is_bool()) dt.code = kDLBool;
    else if (ns.is_unsigned()) dt.code = kDLUInt;
    else if (ns.is_int()) dt.code = kDLInt;
    else if (ns.is_float()) dt.code = kDLFloat;
    else {
        LOGf << "dlpack: unsupported dtype for export:" << ns;
        dt.code = 0; dt.bits = 0;
    }
    return dt;
}

static NanoString dltype_to_ns(DLDataType dt) {
    if (dt.lanes != 1)
        LOGf << "dlpack: vector dtypes (lanes=" << dt.lanes << ") are not supported";
    switch (dt.code) {
        case kDLBool:
            if (dt.bits == 8) return ns_bool;
            break;
        case kDLInt:
            if (dt.bits == 8) return ns_int8;
            if (dt.bits == 16) return ns_int16;
            if (dt.bits == 32) return ns_int32;
            if (dt.bits == 64) return ns_int64;
            break;
        case kDLUInt:
            if (dt.bits == 8) return ns_uint8;
            if (dt.bits == 16) return ns_uint16;
            if (dt.bits == 32) return ns_uint32;
            if (dt.bits == 64) return ns_uint64;
            break;
        case kDLFloat:
            if (dt.bits == 16) return ns_float16;
            if (dt.bits == 32) return ns_float32;
            if (dt.bits == 64) return ns_float64;
            break;
        case kDLBfloat:
            if (dt.bits == 16) return ns_bfloat16;
            break;
        case kDLComplex:
            if (dt.bits == 64) return ns_complex64;
            if (dt.bits == 128) return ns_complex128;
            break;
    }
    LOGf << "dlpack: unsupported DLDataType(code=" << (int)dt.code
         << ", bits=" << (int)dt.bits << ", lanes=" << dt.lanes << ") for import";
    return ns_void;
}

// ---------------------------------------------------------------------
// device <-> DLDevice
// ---------------------------------------------------------------------

// Mirrors the device_of() lambda in VarHolder::copy_into (var_holder.cc):
// Device{} (BackendId::Cpu, 0) for a CPU-resident Var, the real accelerator
// device otherwise.
static Device var_to_backend_device(Var* v) {
    Device d{};
    if (v->allocator && v->allocator->is_cuda())
        d = Device{accelerator_backend_id(), v->device_id < 0 ? 0 : v->device_id};
    return d;
}

static DLDevice var_to_dldevice(VarHolder* v) {
    DLDevice dev{kDLCPU, 0};
    string loc = v->location();
    if (loc == "none") {
        // Not yet materialized: force computation (this is also where the
        // "producer has finished writing" sync guarantee for the accelerator
        // case comes from -- see resolve_export).
        v->sync(true, false);
        loc = v->location();
    }
    if (loc == "cpu") { dev.device_type = kDLCPU; dev.device_id = 0; return dev; }
    if (loc == "device") { dev.device_type = kDLCUDA; dev.device_id = v->device_id(); return dev; }
    LOGf << "dlpack: cannot export a Var with location" << loc
         << "(swapped-to-disk Vars are not supported)";
    return dev;
}

// ---------------------------------------------------------------------
// export: Var -> DLManagedTensor capsule
// ---------------------------------------------------------------------

struct JittorDLManagerCtx {
    VarPtr vp;                    // keeps the underlying storage alive
    std::vector<int64_t> shape;   // DLTensor.shape points into this
    std::vector<int64_t> strides; // DLTensor.strides, empty for row-major
    JittorDLManagerCtx(Var* v) : vp(v) {}
};

static void jittor_dlpack_deleter(DLManagedTensor* self) {
    if (!self) return;
    delete (JittorDLManagerCtx*)self->manager_ctx;
    delete self;
}

// Fires if the capsule is garbage-collected without ever being consumed by
// from_dlpack_capsule (which renames it to "used_dltensor" first) -- the
// standard DLPack idiom (used by PyTorch/NumPy) for "capsule created but
// nobody imported it" cleanup, and the mechanism that makes "deleter called
// exactly once, even if never consumed" true.
static void jittor_dlpack_capsule_destructor(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) return;
    auto* managed = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
    if (managed && managed->deleter) managed->deleter(managed);
}

// Versioned counterparts (DLPack >=0.8's DLManagedTensorVersioned, the
// "current standard" struct per dlpack.h) -- same ownership/deleter idiom as
// the classic pair above, just a distinct capsule name ("dltensor_versioned")
// and struct layout, since consumers that opt into a version via
// __dlpack__(max_version=...) expect that exact ABI back.
static void jittor_dlpack_deleter_versioned(DLManagedTensorVersioned* self) {
    if (!self) return;
    delete (JittorDLManagerCtx*)self->manager_ctx;
    delete self;
}

static void jittor_dlpack_capsule_destructor_versioned(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor_versioned")) return;
    auto* managed = (DLManagedTensorVersioned*)PyCapsule_GetPointer(capsule, "dltensor_versioned");
    if (managed && managed->deleter) managed->deleter(managed);
}

// __dlpack__(copy=True): materialize an independent buffer (same device, a
// real backend-ordered copy -- not an aliasing op like clone_op's
// share_with) so the exported capsule owns memory nobody else can mutate or
// free out from under it. `var` must already be synced/materialized by the
// caller.
static VarPtr make_materialized_copy(Var* var) {
    VarPtr vp(var->shape, var->dtype());
    vp->finish_pending_liveness();
    // Still allocate (rather than leaving mem_ptr null) even when size==0:
    // Jittor's own allocators hand back a valid non-null pointer for a
    // zero-byte request, and the rest of the Var/DLPack machinery (the
    // ASSERT in resolve_export, downstream fetch_sync()) relies on that --
    // an early-return-with-null-mem_ptr here would make an empty Var look
    // "unmaterialized" instead of "genuinely empty".
    Device dev = var_to_backend_device(var);
    Allocator* alloc = dev.backend == BackendId::Cpu
        ? cpu_allocator
        : backend_raw_allocator(dev, BackendMemoryKind::Device);
    Allocation a(alloc, var->size);
    if (var->size)
        backend_copy(a.ptr, dev, var->mem_ptr, dev, var->size, true);
    vp->mem_ptr = a.ptr;
    vp->allocator = a.allocator;
    vp->allocation = a.allocation;
    a.ptr = nullptr;
    return vp;
}

// Shared resolution step for both the classic and versioned export paths:
// sync, resolve device/dtype (can throw -- must happen before any heap
// allocation below, else a rejected export would leak ctx/managed), apply
// copy=True's materialization, and build the ctx that owns the exported
// Var's liveness plus the shape buffer DLTensor.shape points into.
struct ResolvedExport {
    JittorDLManagerCtx* ctx;
    Var* var;
    DLDevice dev;
    DLDataType dtype;
};

// The caller's current accelerator device, restored on the way out. Running
// a Var's pending work selects the device Jittor executes on; a consumer such
// as CuPy inside `with cupy.cuda.Device(1)` then found device 0 current and
// refused its own stream ("created on device 1 does not match"). Jittor
// selects its device again whenever it next runs, so restoring is safe.
struct CallerDeviceGuard {
    int device = -1;
    CallerDeviceGuard() {
        if (accelerator_backend_id() == BackendId::Cpu) return;
        const auto& ops = backend_ops(accelerator_backend_id());
        if (ops.caller_device && ops.restore_caller_device)
            device = ops.caller_device();
    }
    ~CallerDeviceGuard() {
        if (device < 0) return;
        const auto& ops = backend_ops(accelerator_backend_id());
        if (ops.caller_device() != device) ops.restore_caller_device(device);
    }
};

// The consumer's `__dlpack__(stream=...)`, per the DLPack / array API
// contract: None asks for the legacy default stream, so everything Jittor
// queued is waited for (a device sync, as before); -1 asks for no
// synchronization at all; any other positive value is the consumer's CUDA
// stream (1 and 2 being the legacy and per-thread default streams), which is
// made to wait for Jittor's compute stream without blocking the host. 0 is
// ambiguous and the protocol disallows it.
struct ExportStream {
    enum Mode { Sync, NoSync, Wait } mode = Sync;
    int64 handle = 0;
};

static ExportStream parse_export_stream(PyObject* stream) {
    ExportStream r;
    if (!stream || stream == Py_None) return r;
    USER_CHECK(PyLong_Check(stream) && !PyBool_Check(stream))
        << "dlpack: __dlpack__(stream=...) must be an int or None, got"
        << Py_TYPE(stream)->tp_name;
    int overflow = 0;
    long long value = PyLong_AsLongLongAndOverflow(stream, &overflow);
    USER_CHECK(!overflow && !(value == -1 && PyErr_Occurred()))
        << "dlpack: __dlpack__(stream=...) does not fit in int64";
    USER_CHECK(value != 0)
        << "dlpack: __dlpack__(stream=0) is ambiguous and disallowed by DLPack; pass 1 "
           "for the legacy default stream, 2 for the per-thread default stream, or the "
           "stream handle itself";
    USER_CHECK(value >= -1) << "dlpack: __dlpack__(stream=" << value << ") is not a stream";
    r.mode = value == -1 ? ExportStream::NoSync : ExportStream::Wait;
    r.handle = value;
    return r;
}

// Make the consumer's stream wait for everything Jittor queued on `dev`.
static void order_consumer_stream(Device dev, int64 handle) {
    const auto& ops = backend_ops(dev.backend);
    if (!ops.event_create || !ops.event_record || !ops.stream_wait_event
            || !ops.event_destroy) {
        backend_synchronize(dev);
        return;
    }
    BackendEvent done = backend_event(dev);
    ops.event_record(done, backend_stream(dev, BackendStreamKind::Compute));
    ops.stream_wait_event(BackendStream{dev, (void*)(intptr_t)handle}, done);
    ops.event_destroy(done);
}

static ResolvedExport resolve_export(VarHolder* v, bool force_copy, ExportStream stream = {}) {
    // Runs whatever is pending for this Var. Only the default (stream=None)
    // also waits for the device; the other two modes order or skip that
    // below, once the device is known.
    v->sync(stream.mode == ExportStream::Sync, false);
    Var* var = v->var;
    ASSERT(var->mem_ptr || var->num == 0) << "dlpack: exporting an unmaterialized Var";

    DLDevice dev = var_to_dldevice(v);
    DLDataType dtype = ns_to_dltype(var->dtype());
    USER_CHECK(dev.device_type != kDLCPU || stream.mode != ExportStream::Wait)
        << "dlpack: __dlpack__(stream=" << stream.handle << ") names a stream, but this "
           "tensor is on the CPU; pass stream=None";

    // Keeps a materialized-copy Var alive across the ctx construction below;
    // JittorDLManagerCtx takes its own independent reference, so releasing
    // copy_holder at function exit is safe.
    VarPtr copy_holder;
    if (force_copy) {
        if (var->is_contiguous()) {
            copy_holder = make_materialized_copy(var);
        } else {
            // A strided Var's bytes are not its elements in order, so the
            // copy is a dense materialization rather than a memcpy of them.
            VarHolder dense(contiguous_storage(var));
            dense.sync(stream.mode == ExportStream::Sync, false);
            copy_holder = VarPtr(dense.var);
        }
        var = copy_holder.ptr;
        // The copy is stream-ordered; stream=None promises it is complete.
        if (stream.mode == ExportStream::Sync && dev.device_type != kDLCPU)
            backend_synchronize(var_to_backend_device(var));
    }
    if (stream.mode == ExportStream::Wait)
        order_consumer_stream(var_to_backend_device(var), stream.handle);

    auto* ctx = new JittorDLManagerCtx(var);
    ctx->shape.reserve(var->shape.size());
    for (int i = 0; i < var->shape.size(); i++)
        ctx->shape.push_back(var->shape[i]);
    // A strided Var -- a transpose or channels-last view of a storage --
    // exports its own element strides over the same bytes, as torch does,
    // instead of being refused. Row-major storage keeps strides=NULL.
    if (!var->is_contiguous()) {
        ctx->strides.reserve(var->shape.size());
        for (int i = 0; i < var->shape.size(); i++)
            ctx->strides.push_back(var->storage_stride(i));
    }
    return {ctx, var, dev, dtype};
}

static void fill_dl_tensor(DLTensor& t, const ResolvedExport& r) {
    t.data = r.var->mem_ptr;
    t.device = r.dev;
    t.ndim = (int32_t)r.ctx->shape.size();
    t.dtype = r.dtype;
    t.shape = r.ctx->shape.empty() ? nullptr : r.ctx->shape.data();
    t.strides = r.ctx->strides.empty() ? nullptr : r.ctx->strides.data();
    t.byte_offset = 0;
}

static PyObject* to_dlpack_capsule(VarHolder* v, bool force_copy=false, ExportStream stream={}) {
    auto r = resolve_export(v, force_copy, stream);
    auto* managed = new DLManagedTensor();
    managed->manager_ctx = r.ctx;
    managed->deleter = jittor_dlpack_deleter;
    fill_dl_tensor(managed->dl_tensor, r);

    PyObject* capsule = PyCapsule_New(managed, "dltensor", jittor_dlpack_capsule_destructor);
    if (!capsule) {
        jittor_dlpack_deleter(managed);
        return nullptr;
    }
    return capsule;
}

// DLPack >=0.8 "current standard" capsule, produced only when a consumer
// opts in via __dlpack__(max_version=(major,minor>=... )) with major>=1 --
// see VarHolder::dlpack(). Real-world default consumers (NumPy/CuPy's own
// from_dlpack, and torch.from_dlpack against a raw capsule) still get the
// classic capsule from to_dlpack_capsule() above; this path only fires when
// the caller explicitly asked for the versioned ABI.
static PyObject* to_dlpack_capsule_versioned(VarHolder* v, bool force_copy=false, ExportStream stream={}) {
    auto r = resolve_export(v, force_copy, stream);
    auto* managed = new DLManagedTensorVersioned();
    managed->version.major = DLPACK_MAJOR_VERSION;
    managed->version.minor = DLPACK_MINOR_VERSION;
    managed->manager_ctx = r.ctx;
    managed->deleter = jittor_dlpack_deleter_versioned;
    // IS_COPIED accurately reflects make_materialized_copy(): the consumer
    // can treat that buffer as solely theirs. READ_ONLY is never set --
    // Jittor Vars carry no read-only concept to report truthfully.
    managed->flags = force_copy ? DLPACK_FLAG_BITMASK_IS_COPIED : 0;
    fill_dl_tensor(managed->dl_tensor, r);

    PyObject* capsule = PyCapsule_New(managed, "dltensor_versioned", jittor_dlpack_capsule_destructor_versioned);
    if (!capsule) {
        jittor_dlpack_deleter_versioned(managed);
        return nullptr;
    }
    return capsule;
}

PyObject* to_dlpack(VarHolder* v) {
    return to_dlpack_capsule(v);
}

PyObject* VarHolder::dlpack(PyObject* stream, PyObject* max_version, PyObject* dl_device, PyObject* copy) {
    CallerDeviceGuard caller_device;
    // `stream`: see parse_export_stream().
    // `max_version=(major,minor)`: per the DLPack spec, a producer MAY
    // return the versioned DLManagedTensorVersioned capsule once the
    // consumer advertises major>=1 support; otherwise it must fall back to
    // the classic capsule (DLPack's own mandated default -- and what
    // NumPy/CuPy/PyTorch's own from_dlpack() consume when they don't pass
    // max_version at all).
    ExportStream export_stream = parse_export_stream(stream);
    bool want_versioned = false;
    if (max_version && max_version != Py_None) {
        int req_major = 0, req_minor = 0;
        if (!PyArg_ParseTuple(max_version, "ii", &req_major, &req_minor))
            LOGf << "dlpack: __dlpack__(max_version=...) must be a (major, minor) tuple";
        want_versioned = req_major >= 1;
    }
    // copy=True: export a freshly materialized, independently-owned buffer
    // (a real backend-ordered copy, same device) instead of aliasing this
    // Var's own storage -- see make_materialized_copy(). copy=False would
    // mean "the consumer requires zero-copy or must fail"; Jittor's default
    // path is already zero-copy, so False (and None, DLPack's "copy if
    // needed" default) both take the normal aliasing export.
    bool force_copy = copy && copy != Py_None && PyObject_IsTrue(copy);
    if (dl_device && dl_device != Py_None) {
        int req_type = 0, req_id = 0;
        if (!PyArg_ParseTuple(dl_device, "ii", &req_type, &req_id))
            LOGf << "dlpack: __dlpack__(dl_device=...) must be a (device_type, device_id) tuple";
        DLDevice actual = var_to_dldevice(this);
        if (req_type != (int)actual.device_type || req_id != actual.device_id)
            LOGf << "dlpack: __dlpack__(dl_device=...) requesting a device other than "
                 << "where this Var actually lives is not supported; "
                 << "call .migrate_to_device() explicitly first";
    }
    return want_versioned ? to_dlpack_capsule_versioned(this, force_copy, export_stream)
                          : to_dlpack_capsule(this, force_copy, export_stream);
}

PyObject* VarHolder::dlpack_device() {
    DLDevice dev = var_to_dldevice(this);
    return Py_BuildValue("(ii)", (int)dev.device_type, (int)dev.device_id);
}

// ---------------------------------------------------------------------
// import: DLManagedTensor capsule -> Var
// ---------------------------------------------------------------------

// Producer-supplied layout, validated with overflow-checked arithmetic
// before anything reads the buffer or the capsule is consumed. `min_off`/
// `max_off` are the element offsets (relative to data+byte_offset) of the
// lowest and highest element the shape/strides can reach; both are 0 and
// meaningless when numel==0.
struct DLTensorSpan {
    int64 numel = 1;
    int64 min_off = 0;
    int64 max_off = 0;
};

static DLTensorSpan validate_dl_tensor(const DLTensor& t, NanoString dtype) {
    USER_CHECK(t.ndim >= 0 && t.ndim <= NanoVector::max_tensor_rank)
        << "dlpack: ndim=" << t.ndim << " is out of range (Jittor supports 0.."
        << (int)NanoVector::max_tensor_rank << " dims)";
    USER_CHECK(t.ndim == 0 || t.shape != nullptr) << "dlpack: DLTensor.shape is null";
    DLTensorSpan span;
    for (int32_t i = 0; i < t.ndim; i++) {
        USER_CHECK(t.shape[i] >= 0) << "dlpack: negative extent" << t.shape[i] << "on axis" << i;
        USER_CHECK(!__builtin_mul_overflow(span.numel, (int64)t.shape[i], &span.numel))
            << "dlpack: element count overflows int64";
    }
    int64 nbytes;
    USER_CHECK(!__builtin_mul_overflow(span.numel, (int64)dtype.dsize(), &nbytes))
        << "dlpack: byte size overflows int64";
    USER_CHECK(t.byte_offset <= (uint64_t)INT64_MAX) << "dlpack: byte_offset overflows int64";
    if (span.numel == 0) return span;
    int64 contiguous_stride = 1;
    for (int32_t i = t.ndim - 1; i >= 0; i--) {
        int64 stride = t.strides ? t.strides[i] : contiguous_stride;
        int64 extent;
        USER_CHECK(!__builtin_mul_overflow(stride, (int64)t.shape[i] - 1, &extent))
            << "dlpack: stride span overflows int64 on axis" << i;
        USER_CHECK(!__builtin_add_overflow(extent < 0 ? span.min_off : span.max_off, extent,
            extent < 0 ? &span.min_off : &span.max_off))
            << "dlpack: stride span overflows int64 on axis" << i;
        // Only reached for the implicit (null-strides) layout, and bounded by
        // numel, which did not overflow above.
        contiguous_stride *= t.shape[i];
    }
    return span;
}

// Whether a non-contiguous layout can be a Jittor storage view of the same
// bytes: Jittor's strides are non-negative element strides (an offset view
// would be needed for a negative one) that fit a NanoVector.
static bool dl_tensor_strides_aliasable(const DLTensor& t) {
    if (t.strides == nullptr) return true;
    vector<int64> strides(t.ndim);
    for (int32_t i = 0; i < t.ndim; i++) {
        if (t.strides[i] < 0) return false;
        strides[i] = t.strides[i];
    }
    return NanoVector::fits(strides);
}

static bool dl_tensor_is_contiguous(const DLTensor& t) {
    if (t.strides == nullptr) return true;
    // An empty tensor has no bytes, so its strides constrain nothing (CuPy
    // reports non-row-major strides for a (0,3) array).
    for (int32_t i = 0; i < t.ndim; i++)
        if (t.shape[i] == 0) return true;
    int64_t expect = 1;
    for (int32_t i = t.ndim - 1; i >= 0; i--) {
        if (t.shape[i] != 1 && t.strides[i] != expect)
            return false;
        expect *= t.shape[i];
    }
    return true;
}

// Shared capsule unwrap: validates the capsule is live (either flavor),
// resolves version, and returns the DLTensor* plus the two possible managed
// pointers (exactly one of which is non-null) -- used by both the read-only
// peek and the real (consuming) import below, so the "which flavor, which
// version" logic exists in exactly one place. Inspection never consumes an
// unsupported version; a direct import consumes it and calls its deleter.
static DLTensor* unwrap_capsule(PyObject* capsule, bool& versioned,
        DLManagedTensor*& managed, DLManagedTensorVersioned*& managed_v,
        bool consume_unsupported) {
    versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    if (!versioned && !PyCapsule_IsValid(capsule, "dltensor"))
        LOGf << "dlpack: expected a live \"dltensor\" or \"dltensor_versioned\" "
             << "capsule (it may already have been consumed by an earlier "
             << "from_dlpack() call -- a DLPack capsule can only be imported once)";
    managed = nullptr;
    managed_v = nullptr;
    if (versioned) {
        managed_v = (DLManagedTensorVersioned*)PyCapsule_GetPointer(capsule, "dltensor_versioned");
        const uint32_t major = managed_v->version.major;
        if (major != DLPACK_MAJOR_VERSION) {
            // Only the version/deleter prefix has a compatible ABI. Do not
            // inspect dl_tensor, even for a read-only peek. Save the version
            // before a producer deleter can destroy the managed header.
            if (consume_unsupported) {
                if (PyCapsule_SetName(capsule, "used_dltensor_versioned") < 0)
                    LOGf << "dlpack: failed to consume an unsupported-version capsule";
                if (managed_v->deleter) managed_v->deleter(managed_v);
            }
            LOGf << "dlpack: unsupported DLPack major version " << major
                 << " (this build understands major version " << DLPACK_MAJOR_VERSION << ")";
        }
        return &managed_v->dl_tensor;
    }
    managed = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
    return &managed->dl_tensor;
}

// Read-only inspection used by from_dlpack()'s Python wrapper to decide,
// *before* committing to consume the capsule, whether the producer's tensor
// is contiguous (fast, aliasing import below) or strided (Python
// materializes a contiguous copy honoring the strides, then imports THAT via
// the flat_len/elem_offset path below). Must never rename the capsule or
// touch the deleter -- a capsule can still be imported normally afterwards.
PyObject* dlpack_peek(PyObject* capsule) {
    bool versioned; DLManagedTensor* managed; DLManagedTensorVersioned* managed_v;
    DLTensor& t = *unwrap_capsule(capsule, versioned, managed, managed_v, false);
    // The same rank bound as the import: an unchecked ndim sized the tuples
    // below and read that many shape/stride entries.
    USER_CHECK(t.ndim >= 0 && t.ndim <= NanoVector::max_tensor_rank
               && (t.ndim == 0 || t.shape != nullptr))
        << "dlpack: malformed DLTensor (ndim=" << t.ndim << ")";
    PyObject* shape_tuple = PyTuple_New(t.ndim);
    for (int32_t i = 0; i < t.ndim; i++)
        PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLongLong((long long)t.shape[i]));
    PyObject* strides_obj;
    if (t.strides == nullptr) {
        strides_obj = Py_None;
        Py_INCREF(Py_None);
    } else {
        strides_obj = PyTuple_New(t.ndim);
        for (int32_t i = 0; i < t.ndim; i++)
            PyTuple_SET_ITEM(strides_obj, i, PyLong_FromLongLong((long long)t.strides[i]));
    }
    PyObject* aliasable = PyBool_FromLong(dl_tensor_strides_aliasable(t));
    PyObject* result = PyTuple_Pack(3, shape_tuple, strides_obj, aliasable);
    Py_DECREF(shape_tuple);
    Py_DECREF(strides_obj);
    Py_DECREF(aliasable);
    return result;
}

int64 dlpack_consumer_stream(int device_id) {
    CallerDeviceGuard caller_device;
    USER_CHECK(accelerator_backend_id() != BackendId::Cpu)
        << "dlpack: this build has no accelerator stream to order a producer before";
    void* handle = backend_stream(Device{accelerator_backend_id(), device_id},
                                  BackendStreamKind::Compute).handle;
    return handle ? (int64)(intptr_t)handle : 1;
}

VarHolder* from_dlpack_capsule(PyObject* capsule, PyObject* flat_len, int64 elem_offset,
                               bool producer_ordered) {
    CallerDeviceGuard caller_device;
    // Accept either capsule flavor a producer might hand back: the classic
    // "dltensor" (still the default from NumPy/CuPy/PyTorch's own
    // __dlpack__() when the caller doesn't opt into a version) or the
    // "current standard" "dltensor_versioned" (DLManagedTensorVersioned) --
    // see VarHolder::dlpack()/to_dlpack_capsule_versioned() for the export
    // side of the same pair. Only one of these two names can ever be valid
    // on a given capsule.
    bool versioned; DLManagedTensor* managed; DLManagedTensorVersioned* managed_v;
    DLTensor& t = *unwrap_capsule(capsule, versioned, managed, managed_v, true);

    // `flat_len` (a Python int) is set only by from_dlpack()'s own strided-
    // import fallback, which has already decided (via dlpack_peek()) that
    // the producer's tensor is non-contiguous, computed the flattened
    // element span that covers every byte the given strides can reach, and
    // will gather the real (target) shape out of this flat buffer itself
    // with an ordinary jittor advanced-index op afterwards. Importing
    // "shape=[flat_len]" here, with the *original* strides validation
    // skipped, is safe precisely because nothing downstream of this
    // function ever treats this Var as a tensor with the producer's logical
    // shape -- it is only ever a flat, genuinely contiguous 1-D buffer of
    // raw elements, honoring Jittor's real contiguous-only Var invariant.
    // `elem_offset` shifts the imported pointer to the true minimum byte
    // reachable via the strides (which can be negative-offset from
    // `data+byte_offset` for a reversed/negative-stride view), so index 0 of
    // the flat buffer is well-defined.
    // The consumer validates the producer's metadata itself, before any
    // other field is interpreted: a bad shape, stride or flat span must be
    // rejected here, while the capsule is still unconsumed, not discovered
    // later as an out-of-bounds device read.
    NanoString dtype = dltype_to_ns(t.dtype);
    DLTensorSpan span = validate_dl_tensor(t, dtype);

    bool use_flat = flat_len && flat_len != Py_None;
    int64 flat_n = 0;
    if (use_flat) {
        USER_CHECK(PyLong_Check(flat_len)) << "dlpack: flat_len must be an int";
        flat_n = PyLong_AsLongLong(flat_len);
        if (flat_n == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            USER_ERROR << "dlpack: flat_len does not fit in int64";
        }
        USER_CHECK(flat_n >= 0) << "dlpack: flat_len must be non-negative, got" << flat_n;
    }
    // A strided layout is aliased as a storage view of the producer's bytes
    // (zero-copy, as torch imports one). Only a stride Jittor cannot express
    // -- a negative one -- is left to jt.from_dlpack(), which gathers a copy.
    bool strided = !use_flat && span.numel > 0 && !dl_tensor_is_contiguous(t);
    USER_CHECK(!strided || dl_tensor_strides_aliasable(t))
        << "dlpack: this tensor has negative strides, which a Jittor Var cannot alias; "
           "jt.from_dlpack() imports it as a copy";
    if (t.device.device_type != kDLCPU && t.device.device_type != kDLCUDA)
        LOGf << "dlpack: unsupported DLDevice.device_type=" << (int)t.device.device_type
             << " (only kDLCPU and kDLCUDA are supported)";

    if (use_flat) {
        if (flat_n == 0) {
            USER_CHECK(elem_offset == 0) << "dlpack: an empty flat span must have elem_offset=0";
        } else {
            USER_CHECK(span.numel > 0 && elem_offset >= span.min_off
                && elem_offset <= span.max_off && flat_n - 1 <= span.max_off - elem_offset)
                << "dlpack: flat span [" << elem_offset << "," << elem_offset << "+" << flat_n
                << ") is outside the producer's reachable elements ["
                << span.min_off << "," << span.max_off << "]";
        }
    } else {
        USER_CHECK(elem_offset == 0) << "dlpack: elem_offset requires flat_len";
    }
    USER_CHECK(t.data != nullptr || (use_flat ? flat_n : span.numel) == 0)
        << "dlpack: non-empty tensor with a null data pointer";
    int64 offset_bytes;
    USER_CHECK(!__builtin_mul_overflow(elem_offset, (int64)dtype.dsize(), &offset_bytes)
        && !__builtin_add_overflow(offset_bytes, (int64)t.byte_offset, &offset_bytes))
        << "dlpack: data offset overflows int64";
    NanoVector shape;
    if (use_flat)
        shape = NanoVector(flat_n);
    else if (t.ndim > 0)
        shape = NanoVector::make(t.shape, t.ndim);
    else
        shape.clear();

    // Everything that can fail (contiguity, device type, dtype) has already
    // succeeded -- only now do we commit to consuming this capsule. Renaming
    // any earlier would mean a validation failure above leaves the capsule
    // looking "consumed" (jittor_dlpack_capsule_destructor no-ops on it)
    // while nothing ever took ownership of the producer's buffer, permanently
    // leaking it (real accelerator memory, for a CUDA producer).
    if (PyCapsule_SetName(capsule, versioned ? "used_dltensor_versioned" : "used_dltensor") < 0) {
        // Not consumed: the capsule's own destructor still owns the deleter.
        PyErr_Clear();
        LOGf << "dlpack: failed to mark the capsule as consumed";
    }

    // A producer may supply a null deleter. Keep a callable no-op in that
    // case: ForeignAllocator still invokes its release callback exactly once.
    std::function<void()> fire_deleter = versioned
        ? std::function<void()>([managed_v, deleter=managed_v->deleter]() {
            if (deleter) deleter(managed_v);
        })
        : std::function<void()>([managed, deleter=managed->deleter]() {
            if (deleter) deleter(managed);
        });

    // From here on this function owns the producer's deleter. Until an
    // Allocation takes it over, any failure must still release it exactly
    // once, since the renamed capsule's destructor no longer will.
    std::function<void()> release_producer = fire_deleter;
    bool producer_owned = false;
    int target_device_id = t.device.device_type == kDLCPU ? -1 : t.device.device_id;
    try {
        VarPtr vp(shape, dtype);
        vp->finish_pending_liveness();
        void* data = (char*)t.data + offset_bytes;
        // The bytes the view can reach: the logical size for row-major data,
        // the stride span (validated above) for a strided one.
        int64 alias_bytes = vp->size;
        if (strided) {
            vector<int64> strides(t.strides, t.strides + t.ndim);
            vp->set_storage_strides(strides);
            alias_bytes = (span.max_off + 1) * (int64)dtype.dsize();
        }

        Allocation allocation;
        if (t.data == nullptr) {
            // Producers legitimately hand back a null pointer for an empty
            // tensor (CuPy does for a (0,3) array). There are no bytes to
            // alias, so give the Var a genuine zero-byte buffer from the
            // target device's normal allocator stack -- the same one every
            // native empty Var gets, which returns non-null (the raw device
            // allocator deliberately returns null for zero bytes) -- and
            // release the producer right away.
            ASSERT(vp->size == 0);
            Device dev = target_device_id < 0 ? Device{}
                : Device{accelerator_backend_id(), target_device_id};
            Allocation empty(get_allocator(dev, false), 0);
            USER_CHECK(empty.ptr) << "dlpack: allocator returned no storage for an empty tensor";
            std::swap(allocation.ptr, empty.ptr);
            std::swap(allocation.allocation, empty.allocation);
            std::swap(allocation.size, empty.size);
            std::swap(allocation.allocator, empty.allocator);
            producer_owned = true;
            release_producer();
        } else if (target_device_id < 0) {
            make_foreign_allocation(allocation, data, alias_bytes, std::move(fire_deleter));
            producer_owned = true;
        } else {
            make_foreign_allocation(allocation, data, alias_bytes, std::move(fire_deleter),
                &get_foreign_allocator(target_device_id));
            producer_owned = true;
            // Consumer-side half of the stream contract. A producer handed
            // Jittor's compute stream (`dlpack_consumer_stream`) already made
            // that stream wait for its work; otherwise wait for the device.
            if (!producer_ordered)
                backend_synchronize(Device{accelerator_backend_id(), target_device_id});
        }
        vp->mem_ptr = allocation.ptr;
        vp->allocator = allocation.allocator;
        vp->allocation = allocation.allocation;
        allocation.ptr = nullptr;
        allocation.allocator = nullptr;
        allocation.allocation = 0;

        if (target_device_id >= 0) {
            // Explicit placement (mirrors device_copy_op.cc's explicit-backend
            // branch) so later ops route to the correct physical device without
            // depending on the ambient global use_cuda flag, and pair it with
            // _stop_fuse (mirrors VarHolder's own migrate-to-device path) so the
            // fuser cannot silently merge this Var's consumer op with a
            // neighbor targeting a different device.
            vp->placement = TensorPlacement({accelerator_backend_id(), target_device_id});
            vp->device_id = target_device_id;
            vp->set_flag(VarFlags::_stop_fuse);
        }
        return new VarHolder(std::move(vp));
    } catch (...) {
        if (!producer_owned) release_producer();
        throw;
    }
}

} // jittor
