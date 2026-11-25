#include "ops/composite/tvm_ffi_launch_op.h"
#include "bindings/pyjt/dlpack.h"
#include "core/var.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
#include "runtime/tensor_placement.h"
#ifdef HAS_CUDA
#include <array>
#include <atomic>
#include <mutex>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#endif

namespace jittor {
namespace {

using TvmFfiObjectRefFn = int (*)(void*);

#ifdef HAS_CUDA
// Stable TVM-FFI C ABI definitions (from tvm/ffi/c_api.h & c_env_api.h).
enum : int32_t {
    kTVMFFINone = 0,
    kTVMFFIInt = 1,
    kTVMFFIBool = 2,
    kTVMFFIFloat = 3,
    kTVMFFIDLTensorPtr = 7,
    kTVMFFIStaticObjectBegin = 64,
    kTVMFFIError = 67,
};

struct TVMFFIAnyABI {
    int32_t type_index;
    uint32_t zero_padding;
    union {
        int64_t v_int64;
        double v_float64;
        void* v_ptr;
        uint64_t v_uint64;
    };
};
static_assert(sizeof(TVMFFIAnyABI) == 16, "TVMFFIAny ABI must be 16 bytes");

struct TVMFFIObjectABI {
    uint64_t combined_ref_count;
    int32_t type_index;
    uint32_t padding;
    void (*deleter)(void* self, int flags);
};
static_assert(sizeof(TVMFFIObjectABI) == 24, "TVMFFIObject ABI must be 24 bytes");

struct TVMFFIByteArrayABI {
    const char* data;
    size_t size;
};

struct TVMFFIErrorCellABI {
    TVMFFIByteArrayABI kind;
    TVMFFIByteArrayABI message;
    TVMFFIByteArrayABI backtrace;
    void (*update_backtrace)(void* self, const TVMFFIByteArrayABI* backtrace, int32_t update_mode);
    void* cause_chain;
    void* extra_context;
};

using TvmFfiCallFn = int (*)(void*, TVMFFIAnyABI*, int32_t, TVMFFIAnyABI*);
using TvmFfiSetStreamFn = int (*)(int32_t, int32_t, void*, void**);
using TvmFfiErrorMoveFn = void (*)(void**);

static DLDataType var_ns_to_dltype(NanoString ns) {
    DLDataType dt;
    dt.lanes = 1;
    dt.bits = static_cast<uint8_t>(ns.dsize() * 8);
    if (ns.is_complex()) dt.code = kDLComplex;
    else if (ns == ns_bfloat16) dt.code = kDLBfloat;
    else if (ns.is_bool()) dt.code = kDLBool;
    else if (ns.is_unsigned()) dt.code = kDLUInt;
    else if (ns.is_int()) dt.code = kDLInt;
    else if (ns.is_float()) dt.code = kDLFloat;
    else {
        USER_ERROR << "tvm_ffi_launch: unsupported tensor dtype: " << ns;
    }
    return dt;
}

// On a freshly spawned async worker thread, cudaSetDevice() alone may not have
// pushed the CUDA primary context onto the driver stack yet. Ensure a valid
// driver context before invoking a TVM-FFI CUDA module without touching graph
// capture (where a driver context is already current).
static void ensure_cuda_driver_context_if_missing(int device_index) {
#ifndef _WIN32
    using CuCtxGetCurrentFn = int (*)(void**);
    using CuCtxSetCurrentFn = int (*)(void*);
    using CuDeviceGetFn = int (*)(int*, int);
    using CuDevicePrimaryCtxRetainFn = int (*)(void**, int);

    struct Syms {
        CuCtxGetCurrentFn ctx_get = nullptr;
        CuCtxSetCurrentFn ctx_set = nullptr;
        CuDeviceGetFn dev_get = nullptr;
        CuDevicePrimaryCtxRetainFn primary_retain = nullptr;
    };
    static Syms syms;
    static std::once_flag once;
    std::call_once(once, []() {
        void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) lib = RTLD_DEFAULT;
        syms.ctx_get = reinterpret_cast<CuCtxGetCurrentFn>(dlsym(lib, "cuCtxGetCurrent"));
        syms.ctx_set = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(lib, "cuCtxSetCurrent"));
        syms.dev_get = reinterpret_cast<CuDeviceGetFn>(dlsym(lib, "cuDeviceGet"));
        syms.primary_retain = reinterpret_cast<CuDevicePrimaryCtxRetainFn>(
            dlsym(lib, "cuDevicePrimaryCtxRetain"));
    });
    if (!syms.ctx_get || !syms.ctx_set || !syms.dev_get || !syms.primary_retain) return;
    void* current = nullptr;
    if (syms.ctx_get(&current) == 0 && current != nullptr) return;
    if (graph_capture_active()) return;
    if (device_index < 0 || device_index >= 64) return;

    static std::array<std::atomic<void*>, 64> primary_ctxs{};
    static std::mutex ctx_mu;
    void* ctx = primary_ctxs[device_index].load(std::memory_order_acquire);
    if (!ctx) {
        std::lock_guard<std::mutex> lock(ctx_mu);
        ctx = primary_ctxs[device_index].load(std::memory_order_relaxed);
        if (!ctx) {
            int cu_dev = 0;
            if (syms.dev_get(&cu_dev, device_index) == 0
                    && syms.primary_retain(&ctx, cu_dev) == 0 && ctx) {
                primary_ctxs[device_index].store(ctx, std::memory_order_release);
            }
        }
    }
    if (ctx) syms.ctx_set(ctx);
#else
    (void)device_index;
#endif
}

struct TvmFfiStreamGuard {
    TvmFfiSetStreamFn set_stream = nullptr;
    int32_t device_type = kDLCUDA;
    int32_t device_id = 0;
    void* prev_stream = nullptr;
    bool active = false;
    ~TvmFfiStreamGuard() {
        if (active && set_stream)
            set_stream(device_type, device_id, prev_stream, nullptr);
    }
};
#endif

} // namespace

void TvmFfiLaunchOp::parse_validate_and_retain(const vector<uint64>& ffi_handles,
                                               const vector<int>& bound_pairs) {
    USER_CHECK(ffi_handles.size() == 8)
        << "tvm_ffi_launch: ffi_handles must contain 8 elements, got " << ffi_handles.size();
    func_handle = ffi_handles[0];
    mod_handle = ffi_handles[1];
    fn_call = ffi_handles[2];
    fn_set_stream = ffi_handles[3];
    fn_error_move = ffi_handles[4];
    fn_incref = ffi_handles[5];
    fn_decref = ffi_handles[6];
    expected_device = static_cast<int>(ffi_handles[7]);

    USER_CHECK(func_handle && fn_call && fn_set_stream && fn_incref && fn_decref)
        << "tvm_ffi_launch: missing required TVM-FFI C ABI handles";
    USER_CHECK(expected_device >= 0)
        << "tvm_ffi_launch: invalid expected CUDA device " << expected_device;
    USER_CHECK(param_kinds.size() == param_values.size())
        << "tvm_ffi_launch: param_kinds and param_values size mismatch";

    for (size_t i = 0; i < param_kinds.size(); ++i) {
        int kind = param_kinds[i];
        uint64 val = param_values[i];
        if (kind == kInputTensor) {
            USER_CHECK(val < read_buffers.size())
                << "tvm_ffi_launch: input tensor index out of range at param " << i;
        } else if (kind == kOutputTensor) {
            USER_CHECK(val < write_buffers.size())
                << "tvm_ffi_launch: output tensor index out of range at param " << i;
        } else if (kind < kInputTensor || kind > kNone) {
            USER_ERROR << "tvm_ffi_launch: unknown parameter kind " << kind;
        }
    }

    USER_CHECK(bound_pairs.size() % 2 == 0)
        << "tvm_ffi_launch: bound_pairs must have even length";
    if (!bound_pairs.empty()) {
        size_t n_bounds = bound_pairs.size() / 2;
        bound_out_indices.reserve(n_bounds);
        bound_in_indices.reserve(n_bounds);
        for (size_t i = 0; i < n_bounds; ++i) {
            int out_i = bound_pairs[2 * i];
            int in_i = bound_pairs[2 * i + 1];
            USER_CHECK(out_i >= 0 && (size_t)out_i < write_buffers.size()
                    && in_i >= 0 && (size_t)in_i < read_buffers.size())
                << "tvm_ffi_launch: bound pair index out of range";
            bound_out_indices.push_back(out_i);
            bound_in_indices.push_back(in_i);
        }
    }

    // Retain strong C++ references only after all checks pass so constructor
    // exceptions can never leak TVM-FFI object refcounts.
    auto incref = reinterpret_cast<TvmFfiObjectRefFn>(fn_incref);
    USER_CHECK(incref(reinterpret_cast<void*>(func_handle)) == 0)
        << "tvm_ffi_launch: TVMFFIObjectIncRef(func) failed";
    if (mod_handle) {
        if (incref(reinterpret_cast<void*>(mod_handle)) != 0) {
            reinterpret_cast<TvmFfiObjectRefFn>(fn_decref)(reinterpret_cast<void*>(func_handle));
            USER_ERROR << "tvm_ffi_launch: TVMFFIObjectIncRef(mod) failed";
        }
    }
    handles_owned = true;
}

TvmFfiLaunchOp::TvmFfiLaunchOp(
        vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes,
        vector<Var*>&& inputs,
        vector<uint64>&& ffi_handles,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values)
    : read_buffers(inputs),
      param_kinds(move(param_kinds)),
      param_values(move(param_values)) {
    USER_CHECK(!shapes.empty() && shapes.size() == dtypes.size())
        << "tvm_ffi_launch needs matching nonempty output shapes and dtypes";
    USER_CHECK(accelerator_backend_id() == BackendId::Cuda)
        << "tvm_ffi_launch requires CUDA backend support";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);

    const int target_dev = ffi_handles.size() >= 8 ? static_cast<int>(ffi_handles[7]) : 0;
    TensorPlacement zero_input_placement;
    if (read_buffers.empty()) {
        USER_CHECK(target_dev >= 0) << "tvm_ffi_launch: invalid CUDA device " << target_dev;
        zero_input_placement = TensorPlacement({BackendId::Cuda, target_dev});
    }
    write_buffers.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int j = 0; j < shapes[i].size(); ++j)
            USER_CHECK(shapes[i][j] >= 0) << "tvm_ffi_launch requires concrete output shapes";
        Var* out = create_output(shapes[i], dtypes[i]);
        if (read_buffers.empty()) {
            out->placement = zero_input_placement;
            out->device_id = target_dev;
        }
        write_buffers.push_back(out);
    }
    parse_validate_and_retain(ffi_handles, {});
}

TvmFfiLaunchOp::TvmFfiLaunchOp(
        vector<Var*>&& inputs,
        vector<Var*>&& outputs,
        vector<uint64>&& ffi_handles,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& bound_pairs)
    : read_buffers(inputs),
      param_kinds(move(param_kinds)),
      param_values(move(param_values)) {
    USER_CHECK(!outputs.empty()) << "tvm_ffi_launch requires at least one output";
    USER_CHECK(accelerator_backend_id() == BackendId::Cuda)
        << "tvm_ffi_launch requires CUDA backend support";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);

    write_buffers.reserve(outputs.size());
    for (Var* o : outputs) {
        Var* out = create_output(o->shape, o->dtype());
        out->share_with(o);
        write_buffers.push_back(out);
    }
    parse_validate_and_retain(ffi_handles, bound_pairs);
    order_bound_outputs = !bound_out_indices.empty();
}

TvmFfiLaunchOp::~TvmFfiLaunchOp() {
    if (handles_owned && fn_decref) {
        auto decref = reinterpret_cast<TvmFfiObjectRefFn>(fn_decref);
        if (func_handle) decref(reinterpret_cast<void*>(func_handle));
        if (mod_handle) decref(reinterpret_cast<void*>(mod_handle));
    }
}

void TvmFfiLaunchOp::infer_shape() {
    if (!order_bound_outputs) return;
    order_bound_outputs = false;
    vector<Var*> written;
    written.reserve(write_buffers.size());
    for (Var* output : write_buffers)
        if (output->share_src) written.push_back(output->share_src);
    if (!written.empty())
        order_after_readers(this, written);
}

void TvmFfiLaunchOp::run() {
#ifndef HAS_CUDA
    USER_ERROR << "tvm_ffi_launch requires CUDA backend support";
#else
    USER_CHECK(!write_buffers.empty()) << "tvm_ffi_launch requires at least one output";
    Device device = allocation_device(write_buffers[0]->allocator);
    USER_CHECK(device.backend == BackendId::Cuda && device.index == expected_device)
        << "tvm_ffi_launch: expected CUDA device " << expected_device
        << ", got backend " << int(device.backend) << " device " << device.index;
    for (Var* v : read_buffers) {
        Device d = allocation_device(v->allocator);
        USER_CHECK(d.backend == BackendId::Cuda && d.index == expected_device)
            << "tvm_ffi_launch: all input buffers must be on CUDA device " << expected_device;
    }
    for (Var* v : write_buffers) {
        Device d = allocation_device(v->allocator);
        USER_CHECK(d.backend == BackendId::Cuda && d.index == expected_device)
            << "tvm_ffi_launch: all output buffers must be on CUDA device " << expected_device;
    }

    if (!bound_out_indices.empty()) {
        for (size_t i = 0; i < bound_out_indices.size(); ++i) {
            Var* out_v = write_buffers[bound_out_indices[i]];
            Var* in_v = read_buffers[bound_in_indices[i]];
            USER_CHECK(out_v->mem_ptr == in_v->mem_ptr)
                << "tvm_ffi_launch: bound output lost its storage binding";
        }
        if (bound_out_indices.size() > 1) {
            for (size_t left = 0; left < bound_out_indices.size(); ++left) {
                Var* lv = write_buffers[bound_out_indices[left]];
                uintptr_t l0 = reinterpret_cast<uintptr_t>(lv->mem_ptr);
                uintptr_t l1 = l0 + static_cast<uintptr_t>(lv->size);
                for (size_t right = 0; right < left; ++right) {
                    Var* rv = write_buffers[bound_out_indices[right]];
                    uintptr_t r0 = reinterpret_cast<uintptr_t>(rv->mem_ptr);
                    uintptr_t r1 = r0 + static_cast<uintptr_t>(rv->size);
                    USER_CHECK(l1 <= r0 || r1 <= l0)
                        << "tvm_ffi_launch: bound outputs overlap";
                }
            }
        }
    }

    ensure_cuda_driver_context_if_missing(device.index);

    // Count tensors and total dimensions so stack storage covers the common case
    // with zero heap allocation.
    const size_t nargs = param_kinds.size();
    size_t ntensors = 0;
    size_t total_dims = 0;
    for (size_t i = 0; i < nargs; ++i) {
        int kind = param_kinds[i];
        if (kind == kInputTensor) {
            ntensors++;
            total_dims += read_buffers[param_values[i]]->shape.size();
        } else if (kind == kOutputTensor) {
            ntensors++;
            total_dims += write_buffers[param_values[i]]->shape.size();
        }
    }

    constexpr size_t kStackArgs = 64;
    constexpr size_t kStackTensors = 32;
    constexpr size_t kStackDims = 128;

    TVMFFIAnyABI stack_args[kStackArgs];
    DLTensor stack_tensors[kStackTensors];
    int64_t stack_shapes[kStackDims];
    int64_t stack_strides[kStackDims];

    vector<TVMFFIAnyABI> heap_args;
    vector<DLTensor> heap_tensors;
    vector<int64_t> heap_shapes;
    vector<int64_t> heap_strides;

    TVMFFIAnyABI* args = stack_args;
    if (nargs > kStackArgs) {
        heap_args.resize(nargs);
        args = heap_args.data();
    }
    DLTensor* tensors = stack_tensors;
    if (ntensors > kStackTensors) {
        heap_tensors.resize(ntensors);
        tensors = heap_tensors.data();
    }
    int64_t* shapes_buf = stack_shapes;
    int64_t* strides_buf = stack_strides;
    if (total_dims > kStackDims) {
        heap_shapes.resize(total_dims);
        heap_strides.resize(total_dims);
        shapes_buf = heap_shapes.data();
        strides_buf = heap_strides.data();
    }

    const DLDevice dl_dev{kDLCUDA, device.index};
    size_t t_idx = 0;
    size_t dim_off = 0;
    for (size_t i = 0; i < nargs; ++i) {
        int kind = param_kinds[i];
        args[i].zero_padding = 0;
        if (kind == kInputTensor || kind == kOutputTensor) {
            Var* v = kind == kInputTensor
                ? read_buffers[param_values[i]]
                : write_buffers[param_values[i]];
            DLTensor& dt = tensors[t_idx++];
            dt.data = v->mem_ptr;
            dt.device = dl_dev;
            int ndim = v->shape.size();
            dt.ndim = ndim;
            dt.dtype = var_ns_to_dltype(v->dtype());
            dt.byte_offset = 0;
            if (ndim > 0) {
                dt.shape = shapes_buf + dim_off;
                dt.strides = strides_buf + dim_off;
                for (int d = 0; d < ndim; ++d) {
                    shapes_buf[dim_off + d] = v->shape[d];
                    strides_buf[dim_off + d] = v->storage_stride(d);
                }
                dim_off += static_cast<size_t>(ndim);
            } else {
                dt.shape = nullptr;
                dt.strides = nullptr;
            }
            args[i].type_index = kTVMFFIDLTensorPtr;
            args[i].v_ptr = &dt;
        } else if (kind == kInt) {
            args[i].type_index = kTVMFFIInt;
            args[i].v_int64 = static_cast<int64_t>(param_values[i]);
        } else if (kind == kBool) {
            args[i].type_index = kTVMFFIBool;
            args[i].v_int64 = param_values[i] ? 1 : 0;
        } else if (kind == kFloat) {
            args[i].type_index = kTVMFFIFloat;
            args[i].v_uint64 = param_values[i];
        } else if (kind == kNone) {
            args[i].type_index = kTVMFFINone;
            args[i].v_int64 = 0;
        } else {
            USER_ERROR << "tvm_ffi_launch: invalid parameter kind " << kind;
        }
    }

    auto set_stream = reinterpret_cast<TvmFfiSetStreamFn>(fn_set_stream);
    void* compute_stream = backend_stream(device, BackendStreamKind::Compute).handle;
    TvmFfiStreamGuard stream_guard;
    stream_guard.set_stream = set_stream;
    stream_guard.device_type = kDLCUDA;
    stream_guard.device_id = device.index;
    int s_rc = set_stream(kDLCUDA, device.index, compute_stream, &stream_guard.prev_stream);
    USER_CHECK(s_rc == 0) << "tvm_ffi_launch: TVMFFIEnvSetStream failed with code " << s_rc;
    stream_guard.active = true;

    TVMFFIAnyABI result{};
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;

    auto call_fn = reinterpret_cast<TvmFfiCallFn>(fn_call);
    int rc = call_fn(reinterpret_cast<void*>(func_handle), args, static_cast<int32_t>(nargs), &result);

    if (result.type_index >= kTVMFFIStaticObjectBegin && result.v_ptr && fn_decref) {
        reinterpret_cast<TvmFfiObjectRefFn>(fn_decref)(result.v_ptr);
    }

    if (PREDICT_BRANCH_NOT_TAKEN(rc != 0)) {
        string err_msg = "TVMFFIFunctionCall returned error code " + S(rc);
        if (fn_error_move) {
            void* err_obj = nullptr;
            reinterpret_cast<TvmFfiErrorMoveFn>(fn_error_move)(&err_obj);
            if (err_obj) {
                const auto* hdr = reinterpret_cast<const TVMFFIObjectABI*>(err_obj);
                if (hdr->type_index == kTVMFFIError) {
                    const auto* cell = reinterpret_cast<const TVMFFIErrorCellABI*>(
                        reinterpret_cast<const char*>(err_obj) + sizeof(TVMFFIObjectABI));
                    string kind(cell->kind.data ? cell->kind.data : "", cell->kind.size);
                    string message(cell->message.data ? cell->message.data : "", cell->message.size);
                    if (!kind.empty() && !message.empty())
                        err_msg = kind + ": " + message;
                    else if (!message.empty())
                        err_msg = message;
                    else if (!kind.empty())
                        err_msg = kind;
                }
                if (fn_decref)
                    reinterpret_cast<TvmFfiObjectRefFn>(fn_decref)(err_obj);
            }
        }
        USER_ERROR << "tvm_ffi_launch failed: " << err_msg;
    }
#endif
}

VarPtr TvmFfiLaunchOp::grad(Var*, Var*, Var*, int) {
    LOGf << "tvm_ffi_launch has no implicit derivative; define backward with jittor.Function";
    return nullptr;
}

} // namespace jittor
