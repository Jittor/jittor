// jtorch_aten.cu — ALL jittor/CUDA-touching code for the shim (one copy per ext
// .so). Compiled by nvcc. The public torch/extension.h stays jittor/CUDA-free so
// it never drags float3/int3 into extension TUs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <vector>
#include <memory>
#include <map>

#include "common.h"
#include "var.h"
#include "var_holder.h"
#include "executor.h"
#include "ops/op_register.h"
#include "ops/array_op.h"
#include "ops/getitem_op.h"
#include "mem/allocator.h"
#include "pyjt/py_obj_holder.h"
#include "pyjt/py_converter.h"

#include <torch/extension.h>   // interface (forward-declares jittor::VarHolder)

namespace jtorch {

static bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "yes") == 0 ||
           std::strcmp(v, "YES") == 0 || std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
}

static bool env_falsey(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    return std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
           std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "no") == 0 ||
           std::strcmp(v, "NO") == 0 || std::strcmp(v, "off") == 0 ||
           std::strcmp(v, "OFF") == 0;
}

static bool torch_ext_stats_enabled() {
    static const bool enabled = env_truthy("JITTOR_TORCH_EXT_STATS") ||
                                env_truthy("JITTOR_TORCH_EXT_PROFILE");
    return enabled;
}

#define JT_STRINGIFY_IMPL(x) #x
#define JT_STRINGIFY(x) JT_STRINGIFY_IMPL(x)
#ifdef TORCH_EXTENSION_NAME
static const char* torch_ext_module_name() { return JT_STRINGIFY(TORCH_EXTENSION_NAME); }
#elif defined(JTORCH_EXTENSION_MODULE_NAME)
static const char* torch_ext_module_name() { return JT_STRINGIFY(JTORCH_EXTENSION_MODULE_NAME); }
#else
static const char* torch_ext_module_name() { return "unknown"; }
#endif

struct TorchExtStats {
    std::atomic<long long> input_total{0};
    std::atomic<long long> input_clone_count{0};
    std::atomic<long long> input_clone_bytes{0};
    std::atomic<long long> input_readonly_borrow_count{0};
    std::atomic<long long> input_readonly_borrow_bytes{0};
    std::atomic<long long> input_mutable_borrow_count{0};
    std::atomic<long long> input_mutable_borrow_bytes{0};
    std::atomic<long long> input_global_borrow_count{0};
    std::atomic<long long> input_global_borrow_bytes{0};
    std::atomic<long long> input_empty_count{0};
    std::atomic<long long> force_cpu_count{0};
    std::atomic<long long> force_cpu_bytes{0};
    std::atomic<long long> direct_readonly_count{0};
    std::atomic<long long> direct_readonly_bytes{0};
    std::atomic<long long> output_count{0};
    std::atomic<long long> output_bytes{0};
    std::atomic<long long> mutable_commit_count{0};
    std::atomic<long long> mutable_commit_bytes{0};
    std::atomic<long long> data_ptr_fast_count{0};
    std::atomic<long long> data_ptr_fast_bytes{0};
    std::atomic<long long> data_ptr_sync_count{0};
    std::atomic<long long> data_ptr_sync_bytes{0};
    std::atomic<long long> metadata_fast_count{0};
    std::atomic<long long> metadata_sync_count{0};
    std::atomic<long long> native_scalar_index_count{0};
    std::atomic<long long> host_scalar_index_count{0};
    std::atomic<long long> host_scalar_index_bytes{0};
    std::atomic<long long> dtype_view_native_count{0};
    std::atomic<long long> dtype_view_native_bytes{0};
    std::atomic<long long> dtype_view_host_count{0};
    std::atomic<long long> dtype_view_host_bytes{0};
    std::atomic<long long> host_fallback_copy_count{0};
    std::atomic<long long> host_fallback_copy_bytes{0};
    std::atomic<long long> full_memset_count{0};
    std::atomic<long long> full_memset_bytes{0};
};

static TorchExtStats& torch_ext_stats() {
    static TorchExtStats* stats = new TorchExtStats();
    return *stats;
}

static inline void stat_add(std::atomic<long long>& counter, long long value = 1) {
    if (__builtin_expect(torch_ext_stats_enabled(), 0))
        counter.fetch_add(value, std::memory_order_relaxed);
}

static inline void stat_add_if(bool enabled, std::atomic<long long>& counter, long long value = 1) {
    if (enabled)
        counter.fetch_add(value, std::memory_order_relaxed);
}

static inline long long vh_nbytes_for_stats(jittor::VarHolder* vh) {
    if (!vh || !vh->var)
        return 0;
    return (long long)vh->var->num * (long long)vh->var->dtype().dsize();
}

#define JT_STAT_LOAD(name) stats.name.load(std::memory_order_relaxed)
struct TorchExtStatsReporter {
    ~TorchExtStatsReporter() {
        if (!torch_ext_stats_enabled())
            return;
        TorchExtStats& stats = torch_ext_stats();
        long long activity =
            JT_STAT_LOAD(input_total) + JT_STAT_LOAD(direct_readonly_count) +
            JT_STAT_LOAD(output_count) + JT_STAT_LOAD(data_ptr_fast_count) +
            JT_STAT_LOAD(data_ptr_sync_count) + JT_STAT_LOAD(host_fallback_copy_count);
        if (!activity)
            return;
        const char* path = std::getenv("JITTOR_TORCH_EXT_STATS_FILE");
        FILE* fp = (path && *path) ? std::fopen(path, "a") : stderr;
        if (!fp)
            fp = stderr;
        std::fprintf(fp,
            "[jtorch-ext-stats] module=%s "
            "input_total=%lld input_clone=%lld input_clone_bytes=%lld "
            "readonly_borrow=%lld readonly_borrow_bytes=%lld "
            "mutable_borrow=%lld mutable_borrow_bytes=%lld "
            "global_borrow=%lld global_borrow_bytes=%lld input_empty=%lld "
            "force_cpu=%lld force_cpu_bytes=%lld "
            "direct_readonly=%lld direct_readonly_bytes=%lld "
            "output=%lld output_bytes=%lld mutable_commit=%lld mutable_commit_bytes=%lld "
            "data_ptr_fast=%lld data_ptr_fast_bytes=%lld "
            "data_ptr_sync=%lld data_ptr_sync_bytes=%lld "
            "metadata_fast=%lld metadata_sync=%lld "
            "native_scalar_index=%lld host_scalar_index=%lld host_scalar_index_bytes=%lld "
            "dtype_view_native=%lld dtype_view_native_bytes=%lld "
            "dtype_view_host=%lld dtype_view_host_bytes=%lld "
            "host_fallback_copy=%lld host_fallback_copy_bytes=%lld "
            "full_memset=%lld full_memset_bytes=%lld\n",
            torch_ext_module_name(),
            JT_STAT_LOAD(input_total), JT_STAT_LOAD(input_clone_count), JT_STAT_LOAD(input_clone_bytes),
            JT_STAT_LOAD(input_readonly_borrow_count), JT_STAT_LOAD(input_readonly_borrow_bytes),
            JT_STAT_LOAD(input_mutable_borrow_count), JT_STAT_LOAD(input_mutable_borrow_bytes),
            JT_STAT_LOAD(input_global_borrow_count), JT_STAT_LOAD(input_global_borrow_bytes),
            JT_STAT_LOAD(input_empty_count),
            JT_STAT_LOAD(force_cpu_count), JT_STAT_LOAD(force_cpu_bytes),
            JT_STAT_LOAD(direct_readonly_count), JT_STAT_LOAD(direct_readonly_bytes),
            JT_STAT_LOAD(output_count), JT_STAT_LOAD(output_bytes),
            JT_STAT_LOAD(mutable_commit_count), JT_STAT_LOAD(mutable_commit_bytes),
            JT_STAT_LOAD(data_ptr_fast_count), JT_STAT_LOAD(data_ptr_fast_bytes),
            JT_STAT_LOAD(data_ptr_sync_count), JT_STAT_LOAD(data_ptr_sync_bytes),
            JT_STAT_LOAD(metadata_fast_count), JT_STAT_LOAD(metadata_sync_count),
            JT_STAT_LOAD(native_scalar_index_count),
            JT_STAT_LOAD(host_scalar_index_count), JT_STAT_LOAD(host_scalar_index_bytes),
            JT_STAT_LOAD(dtype_view_native_count), JT_STAT_LOAD(dtype_view_native_bytes),
            JT_STAT_LOAD(dtype_view_host_count), JT_STAT_LOAD(dtype_view_host_bytes),
            JT_STAT_LOAD(host_fallback_copy_count), JT_STAT_LOAD(host_fallback_copy_bytes),
            JT_STAT_LOAD(full_memset_count), JT_STAT_LOAD(full_memset_bytes));
        if (fp != stderr)
            std::fclose(fp);
    }
};
static TorchExtStatsReporter torch_ext_stats_reporter;
#undef JT_STAT_LOAD

static bool torch_ext_sync_data_ptr_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_SYNC_BOUNDARY"))
        return true;
    if (env_falsey("JITTOR_TORCH_EXT_ASYNC_DATA_PTR"))
        return true;
    return env_truthy("JITTOR_TORCH_EXT_SYNC_DATA_PTR");
}

static bool torch_ext_sync_return_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_SYNC_BOUNDARY"))
        return true;
    if (env_falsey("JITTOR_TORCH_EXT_ASYNC_RETURN"))
        return true;
    return env_truthy("JITTOR_TORCH_EXT_SYNC_RETURN");
}

static bool torch_ext_borrow_inputs_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_SYNC_BOUNDARY") ||
        env_truthy("JITTOR_TORCH_EXT_COPY_INPUTS") ||
        env_falsey("JITTOR_TORCH_EXT_UNSAFE_BORROW_INPUTS") ||
        env_falsey("JITTOR_TORCH_EXT_BORROW_INPUTS"))
        return false;
    return env_truthy("JITTOR_TORCH_EXT_UNSAFE_BORROW_INPUTS") ||
           env_truthy("JITTOR_TORCH_EXT_BORROW_INPUTS");
}

bool pyvar_ext_borrow_inputs_enabled() {
    return torch_ext_borrow_inputs_enabled();
}

static bool torch_ext_fast_metadata_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_SYNC_BOUNDARY") ||
        env_truthy("JITTOR_TORCH_EXT_SYNC_METADATA") ||
        env_falsey("JITTOR_TORCH_EXT_UNSAFE_FAST_METADATA") ||
        env_falsey("JITTOR_TORCH_EXT_FAST_METADATA"))
        return false;
    return env_truthy("JITTOR_TORCH_EXT_UNSAFE_FAST_METADATA") ||
           env_truthy("JITTOR_TORCH_EXT_FAST_METADATA");
}

static bool torch_ext_native_scalar_index_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_HOST_INDEX"))
        return false;
    return !env_falsey("JITTOR_TORCH_EXT_NATIVE_INDEX");
}

static bool torch_ext_dtype_view_native_enabled() {
    return !env_falsey("JITTOR_TORCH_EXT_DTYPE_VIEW_NATIVE");
}

static void sync_for_storage(jittor::VarHolder* vh) {
    vh->sync(/*device_sync=*/false, /*weak_sync=*/false);
}

static void sync_for_data_ptr(jittor::VarHolder* vh) {
    vh->sync(/*device_sync=*/torch_ext_sync_data_ptr_enabled(),
             /*weak_sync=*/false);
}

// --------- dtype mapping ----------------------------------------------------
static const char* scalar_to_jt(ScalarType s) {
    switch (s) {
        case ScalarType::Byte:   return "uint8";
        case ScalarType::Char:   return "int8";
        case ScalarType::Short:  return "int16";
        case ScalarType::Int:    return "int32";
        case ScalarType::Long:   return "int64";
        case ScalarType::Half:   return "float16";
        case ScalarType::BFloat16:return "bfloat16";
        case ScalarType::Float:  return "float32";
        case ScalarType::Double: return "float64";
        case ScalarType::Bool:   return "bool";
        case ScalarType::UInt16: return "uint16";
        case ScalarType::UInt32: return "uint32";
        case ScalarType::UInt64: return "uint64";
        default: throw std::runtime_error("jtorch: unsupported ScalarType");
    }
}
static int64_t scalar_size(ScalarType s) {
    switch (s) {
        case ScalarType::Byte:
        case ScalarType::Char:
        case ScalarType::Bool:
            return 1;
        case ScalarType::Short:
        case ScalarType::Half:
        case ScalarType::BFloat16:
        case ScalarType::UInt16:
            return 2;
        case ScalarType::Int:
        case ScalarType::Float:
        case ScalarType::UInt32:
            return 4;
        case ScalarType::Long:
        case ScalarType::Double:
        case ScalarType::UInt64:
            return 8;
        default:
            return 0;
    }
}
ScalarType detail::name_to_scalar(const char* n) {
    std::string s(n);
    if (s=="uint8")  return ScalarType::Byte;
    if (s=="int8")   return ScalarType::Char;
    if (s=="int16")  return ScalarType::Short;
    if (s=="int32")  return ScalarType::Int;
    if (s=="int64")  return ScalarType::Long;
    if (s=="float16")return ScalarType::Half;
    if (s=="bfloat16")return ScalarType::BFloat16;
    if (s=="float32")return ScalarType::Float;
    if (s=="float64")return ScalarType::Double;
    if (s=="bool")   return ScalarType::Bool;
    if (s=="uint16") return ScalarType::UInt16;
    if (s=="uint32") return ScalarType::UInt32;
    if (s=="uint64") return ScalarType::UInt64;
    throw std::runtime_error("jtorch: unknown jittor dtype " + s);
}
static jittor::NanoString to_ns(ScalarType s) { return jittor::NanoString(scalar_to_jt(s)); }
static jittor::NanoVector to_nv(const std::vector<int64_t>& s) {
    jittor::NanoVector nv; for (auto d : s) nv.push_back(d); return nv;
}
static jittor::NanoVector to_nv(const IntArrayRef& s) { return to_nv(s.v); }

// --------- Var construction via jittor's "empty" op -------------------------
// jittor's "empty" allocates per the global use_cuda flag (=> GPU here). For a
// CPU-resident tensor (torch::empty(..., device=kCPU) or empty_like(cpu_tensor))
// we migrate it to the host allocator so its data_ptr is a real host pointer —
// this is what the extensions' *_cpu kernels require.
static jittor::VarHolder* make_empty_vh(const jittor::NanoVector& shape, ScalarType st) {
    static auto maker = jittor::get_op_info("empty")
        .get_constructor<jittor::VarPtr, jittor::NanoVector, jittor::NanoString>();
    jittor::VarPtr vp = maker(shape, to_ns(st));
    return new jittor::VarHolder(std::move(vp));
}
static jittor::VarHolder* make_copy_vh(jittor::Var* src) {
    static auto maker = jittor::get_op_info("copy")
        .get_constructor<jittor::VarPtr, jittor::Var*>();
    jittor::VarPtr vp = maker(src);
    return new jittor::VarHolder(std::move(vp));
}
static jittor::VarHolder* make_reinterpret_view_vh(jittor::Var* src, const jittor::NanoVector& shape, ScalarType st) {
    static auto maker = jittor::get_op_info("reinterpret_view")
        .get_constructor<jittor::VarPtr, jittor::Var*, jittor::NanoVector, jittor::NanoString>();
    jittor::VarPtr vp = maker(src, shape, to_ns(st));
    return new jittor::VarHolder(std::move(vp));
}
static jittor::VarHolder* make_var(const jittor::NanoVector& shape, ScalarType st, bool cpu) {
    jittor::VarHolder* vh = make_empty_vh(shape, st);
    if (cpu) {
        sync_for_storage(vh);
#ifdef HAS_CUDA
        jittor::migrate_to_cpu(vh->var, jittor::exe.allocator);
#endif
    }
    return vh;
}
static bool opt_is_cpu(const TensorOptions& o) { return o.has_device_ && o.device_ == DeviceType::CPU; }

// Build a properly graph-tracked jittor Var from a HOST buffer (jittor's "array"
// op copies the bytes and owns them). Unlike empty()+raw-cudaMemcpy, the Var's
// value survives a later .item()/.numpy()/sync (which may otherwise re-run a
// lazy "empty" op and read fresh uninitialised memory). Used for op results we
// compute on the host (argsort indices, cumsum).
static Tensor var_from_host(const void* host, const jittor::NanoVector& shape, ScalarType st) {
    static auto maker = jittor::get_op_info("array")
        .get_constructor<jittor::VarPtr, const void*, jittor::NanoVector, jittor::NanoString>();
    jittor::VarPtr vp = maker(host, shape, to_ns(st));
    return detail::adopt(new jittor::VarHolder(std::move(vp)), true);
}

// Like var_from_host but, when `to_cuda`, migrates the result onto the GPU so its
// device-residency matches a CUDA source (jittor's "array" op produces a
// host-resident Var until consumed; callers that hand the Var straight to a CUDA
// consumer — e.g. a triton kernel launched via the bridge, which checks
// is_cuda — need it resident on the device). Mirrors jittor's own
// load-path migrate (`migrate_to_gpu(var, get_allocator())`). Stays settled
// (graph-tracked array op) so a later .item()/sync does not re-run a lazy empty.
static Tensor var_from_host_dev(const void* host, const jittor::NanoVector& shape,
                                ScalarType st, bool to_cuda) {
    Tensor t = var_from_host(host, shape, st);
#ifdef HAS_CUDA
    if (to_cuda) {
        sync_for_storage(t._vh());
        jittor::migrate_to_gpu(t._vh()->var, jittor::get_allocator());
    } else {
        sync_for_storage(t._vh());
        jittor::migrate_to_cpu(t._vh()->var, jittor::exe.allocator);
    }
#endif
    return t;
}

// --------- detail bridge helpers --------------------------------------------
namespace detail {

void data_ptrs(std::initializer_list<Tensor> tensors, void** out) {
    std::vector<jittor::VarHolder*> vhs;
    std::vector<size_t> pending_indices;
    vhs.reserve(tensors.size());
    pending_indices.reserve(tensors.size());
    bool collect_stats = torch_ext_stats_enabled();
    bool device_sync = torch_ext_sync_data_ptr_enabled();
    size_t i = 0;
    for (const auto& t : tensors) {
        if (!t.defined()) {
            out[i++] = nullptr;
            continue;
        }
        jittor::VarHolder* vh = t._vh();
        if (!vh || !vh->var || vh->var->num == 0) {
            out[i++] = nullptr;
            continue;
        }
        long long nbytes = collect_stats ? vh_nbytes_for_stats(vh) : 0;
        if (!device_sync && vh->var->mem_ptr && vh->var->allocator) {
            stat_add_if(collect_stats, torch_ext_stats().data_ptr_fast_count);
            stat_add_if(collect_stats, torch_ext_stats().data_ptr_fast_bytes, nbytes);
            out[i++] = vh->var->mem_ptr;
            continue;
        }
        stat_add_if(collect_stats, torch_ext_stats().data_ptr_sync_count);
        stat_add_if(collect_stats, torch_ext_stats().data_ptr_sync_bytes, nbytes);
        vhs.push_back(vh);
        pending_indices.push_back(i++);
    }
    if (!vhs.empty())
        jittor::sync(vhs, device_sync, false);
    for (size_t j = 0; j < vhs.size(); ++j) {
        jittor::VarHolder* vh = vhs[j];
        size_t index = pending_indices[j];
        if (vh->var->allocator && vh->var->allocator->is_cuda()) {
            out[index] = vh->var->mem_ptr;
        } else {
#ifdef HAS_CUDA
            jittor::migrate_to_cpu(vh->var, jittor::exe.allocator);
#endif
            out[index] = vh->var->mem_ptr;
        }
    }
}

void* vh_device_ptr(jittor::VarHolder* vh) {
    // torch returns a NULL data_ptr for an empty (0-element) tensor, and exts
    // dispatch on it: diff-gaussian-rasterization passes empty cov3D_precomp /
    // colors_precomp tensors and branches `if (cov3D_precomp != nullptr)` /
    // `if (colors_precomp == nullptr)` to pick precomputed-vs-(SH/scale) paths.
    // A jittor empty Var has a non-null mem_ptr, which would silently select the
    // wrong (garbage precomputed) branch -> degenerate covariance + black image.
    // Mirror torch: 0-element tensor -> nullptr.
    if (vh->var->num == 0) return nullptr;
    // Return the pointer at the Var's CURRENT location (torch semantics):
    // CUDA var -> device ptr, CPU var -> host ptr. Do NOT force-migrate to GPU,
    // else *_cpu kernels would read a device pointer as host memory and segfault.
    // data_ptr only needs the Jittor graph materialized and queued. Extensions
    // use stream 0 through the c10 stream shim, so their kernels are ordered
    // after Jittor stream-0 work without a global device barrier. Set
    // JITTOR_TORCH_EXT_SYNC_DATA_PTR=1 or JITTOR_TORCH_EXT_SYNC_BOUNDARY=1 for
    // conservative debugging.
    bool collect_stats = torch_ext_stats_enabled();
    long long nbytes = collect_stats ? vh_nbytes_for_stats(vh) : 0;
    if (!torch_ext_sync_data_ptr_enabled() && vh->var->mem_ptr && vh->var->allocator) {
        stat_add_if(collect_stats, torch_ext_stats().data_ptr_fast_count);
        stat_add_if(collect_stats, torch_ext_stats().data_ptr_fast_bytes, nbytes);
        return vh->var->mem_ptr;
    }
    stat_add_if(collect_stats, torch_ext_stats().data_ptr_sync_count);
    stat_add_if(collect_stats, torch_ext_stats().data_ptr_sync_bytes, nbytes);
    sync_for_data_ptr(vh);
    if (vh->var->allocator && vh->var->allocator->is_cuda())
        return vh->var->mem_ptr;
#ifdef HAS_CUDA
    jittor::migrate_to_cpu(vh->var, jittor::exe.allocator);
#endif
    return vh->var->mem_ptr;
}
int64_t vh_ndim(jittor::VarHolder* vh) { return vh->var->shape.size(); }
int64_t vh_size(jittor::VarHolder* vh, int64_t d) {
    int64_t nd = vh->var->shape.size(); if (d < 0) d += nd; return vh->var->shape[d];
}
void vh_shape(jittor::VarHolder* vh, std::vector<int64_t>& out) {
    auto& sh = vh->var->shape; for (int i = 0; i < (int)sh.size(); i++) out.push_back(sh[i]);
}
int64_t vh_numel(jittor::VarHolder* vh) { return vh->var->num; }
int64_t vh_dsize(jittor::VarHolder* vh) { return vh->var->dtype().dsize(); }
const char* vh_dtype_name(jittor::VarHolder* vh) { return vh->var->dtype().to_cstring(); }
bool vh_is_cuda(jittor::VarHolder* vh) {
    if (vh->var->allocator && vh->var->mem_ptr)
        return vh->var->allocator->is_cuda();
    if (torch_ext_fast_metadata_enabled() && vh->var->allocator) {
        stat_add(torch_ext_stats().metadata_fast_count);
        return vh->var->allocator->is_cuda();
    }
    stat_add(torch_ext_stats().metadata_sync_count);
    sync_for_data_ptr(vh);
    return vh->var->allocator && vh->var->allocator->is_cuda();
}
bool vh_allocator_is_cuda(jittor::VarHolder* vh) {
    return vh && vh->var && vh->var->allocator && vh->var->allocator->is_cuda();
}
int vh_device_type(jittor::VarHolder* vh) {
    // Report the Var's CURRENT residency without migrating (torch::Tensor::device()).
    if (vh->var->allocator && vh->var->mem_ptr)
        return vh->var->allocator->is_cuda() ? 1 : 0;
    if (torch_ext_fast_metadata_enabled() && vh->var->allocator) {
        stat_add(torch_ext_stats().metadata_fast_count);
        return vh->var->allocator->is_cuda() ? 1 : 0;
    }
    stat_add(torch_ext_stats().metadata_sync_count);
    sync_for_data_ptr(vh);
    return (vh->var->allocator && vh->var->allocator->is_cuda()) ? 1 : 0;
}
double vh_item_double(jittor::VarHolder* vh) {
    jittor::ItemData d = vh->item(); std::string n = d.dtype.to_cstring();
    if (n == "float32") { float f; std::memcpy(&f, &d.data, 4); return f; }
    if (n == "float64") { double f; std::memcpy(&f, &d.data, 8); return f; }
    return (double)d.data;
}
int64_t vh_item_int(jittor::VarHolder* vh) { return vh->item().data; }

static void _delete_vh(jittor::VarHolder* p) { delete p; }
static void _noop_vh(jittor::VarHolder*) {}

Tensor adopt(jittor::VarHolder* vh, bool owns) {
    std::shared_ptr<jittor::VarHolder> sp(vh, owns ? &_delete_vh : &_noop_vh);
    return Tensor(sp);
}
jittor::VarHolder* clone_holder(jittor::VarHolder* vh) { return new jittor::VarHolder(vh->var); }

bool is_jittor_var(void* obj) {
    return Py_TYPE((PyObject*)obj) == &jittor::PyjtVarHolder.ht_type;
}
bool pyvar_is_ext_mutable(void* obj) {
    PyObject* pyobj = (PyObject*)obj;
    if (!pyobj || Py_TYPE(pyobj) != &jittor::PyjtVarHolder.ht_type)
        return false;
    PyObject* attr = PyObject_GetAttrString(pyobj, "_jittor_torch_ext_mutable");
    if (!attr) {
        PyErr_Clear();
        return false;
    }
    int ok = PyObject_IsTrue(attr);
    Py_DECREF(attr);
    return ok == 1;
}
static bool pyvar_flag(void* obj, const char* name) {
    PyObject* pyobj = (PyObject*)obj;
    if (!pyobj || Py_TYPE(pyobj) != &jittor::PyjtVarHolder.ht_type)
        return false;
    PyObject* attr = PyObject_GetAttrString(pyobj, name);
    if (!attr) {
        PyErr_Clear();
        return false;
    }
    int ok = PyObject_IsTrue(attr);
    Py_DECREF(attr);
    return ok == 1;
}
bool pyvar_is_ext_readonly_borrow(void* obj) {
    return pyvar_flag(obj, "_jittor_torch_ext_readonly_borrow");
}
static bool pyvar_force_cpu(void* obj) {
    return pyvar_flag(obj, "_jittor_torch_force_cpu");
}
void commit_tensor_to_pyvar(void* obj, const Tensor& t) {
    if (!obj || !t.defined())
        return;
#ifdef HAS_CUDA
    if (torch_ext_sync_return_enabled())
        cudaDeviceSynchronize();
#endif
    PyObject* pyobj = (PyObject*)obj;
    if (Py_TYPE(pyobj) != &jittor::PyjtVarHolder.ht_type)
        return;
    bool collect_stats = torch_ext_stats_enabled();
    stat_add_if(collect_stats, torch_ext_stats().mutable_commit_count);
    if (collect_stats && t.defined())
        stat_add_if(collect_stats, torch_ext_stats().mutable_commit_bytes,
                    (long long)t.numel() * t.element_size());
    jittor::VarHolder* dst = jittor::from_py_object<jittor::VarHolder*>(pyobj);
    Tensor settled = t.clone();
    jittor::VarHolder* src = settled._vh();
    if (!src || !src->var || dst->var == src->var)
        return;

    jittor::Var* old_var = dst->var;
    jittor::Var* new_var = src->var;

    // Match VarHolder::assign's attribute preservation, but transfer ownership
    // from the temporary settled holder to the original Python holder.  Calling
    // assign() with a stack/temporary VarHolder is unsafe: the temporary holder's
    // destructor would later unlink/release the Var now owned by `dst`.
    new_var->name = std::move(old_var->name);
    if (old_var->is_stop_grad())
        new_var->set_stop_grad();
    if (old_var->flags.get(jittor::NodeFlags::_stop_fuse))
        new_var->flags.set(jittor::NodeFlags::_stop_fuse);
    if (old_var->flag(jittor::VarFlags::_explicit_requires_grad))
        new_var->set_flag(jittor::VarFlags::_explicit_requires_grad);

    src->release_from_holders();  // remove temp holder from hold_vars; keep liveness
    src->var = nullptr;           // transfer the temp holder's liveness to dst
    dst->release_holder();
    old_var->release_both_liveness();
    dst->var = new_var;
    dst->own_holder();
}
Tensor tensor_from_pyvar(void* obj) {
    jittor::VarHolder* vh = jittor::from_py_object<jittor::VarHolder*>((PyObject*)obj);
    Tensor borrowed = adopt(vh, /*owns=*/false);    // python owns it; we borrow
    bool collect_stats = torch_ext_stats_enabled();
    long long nbytes = (collect_stats && borrowed.defined()) ? (long long)borrowed.numel() * borrowed.element_size() : 0;
    stat_add_if(collect_stats, torch_ext_stats().input_total);
    if (pyvar_force_cpu(obj) && borrowed.defined()) {
        stat_add_if(collect_stats, torch_ext_stats().force_cpu_count);
        stat_add_if(collect_stats, torch_ext_stats().force_cpu_bytes, nbytes);
        if (pyvar_is_ext_readonly_borrow(obj)) {
            stat_add_if(collect_stats, torch_ext_stats().input_readonly_borrow_count);
            stat_add_if(collect_stats, torch_ext_stats().input_readonly_borrow_bytes, nbytes);
            return borrowed;
        }
        if (pyvar_is_ext_mutable(obj)) {
            stat_add_if(collect_stats, torch_ext_stats().input_mutable_borrow_count);
            stat_add_if(collect_stats, torch_ext_stats().input_mutable_borrow_bytes, nbytes);
            return borrowed;
        }
        jittor::NanoVector nv = to_nv(borrowed.sizes());
        if (borrowed.numel() == 0)
            return adopt(make_var(nv, borrowed.scalar_type(), /*cpu=*/true), true);
        int64_t nbytes = borrowed.numel() * detail::vh_dsize(borrowed._vh());
        if (!detail::vh_is_cuda(borrowed._vh()))
            return var_from_host_dev(detail::vh_device_ptr(borrowed._vh()), nv,
                                     borrowed.scalar_type(), false);
        std::unique_ptr<char[]> host(new char[nbytes]);
        cudaMemcpy(host.get(), detail::vh_device_ptr(borrowed._vh()), nbytes,
                   cudaMemcpyDeviceToHost);
        return var_from_host_dev(host.get(), nv, borrowed.scalar_type(), false);
    }
    if (pyvar_is_ext_readonly_borrow(obj)) {
        stat_add_if(collect_stats, torch_ext_stats().input_readonly_borrow_count);
        stat_add_if(collect_stats, torch_ext_stats().input_readonly_borrow_bytes, nbytes);
        return borrowed;
    }
    if (pyvar_is_ext_mutable(obj)) {
        stat_add_if(collect_stats, torch_ext_stats().input_mutable_borrow_count);
        stat_add_if(collect_stats, torch_ext_stats().input_mutable_borrow_bytes, nbytes);
        return borrowed;
    }
    if (torch_ext_borrow_inputs_enabled()) {
        stat_add_if(collect_stats, torch_ext_stats().input_global_borrow_count);
        stat_add_if(collect_stats, torch_ext_stats().input_global_borrow_bytes, nbytes);
        return borrowed;
    }
    // BAKE the input into a SETTLED jittor "array" Var before the ext's kernel
    // reads it. A lazy/intermediate input (e.g. fused-ssim's rendered image =
    // clamp(rasterizer_output), or any value computed several ops before this
    // ext call) lives in a jittor buffer that the executor may recycle/alias —
    // notably the autograd-TAPED input (not a hold_var) whose block the ext's own
    // torch::zeros_like outputs can reuse, so the async kernel reads corrupted
    // data -> intermittent NaN/illegal-address. A settled array Var is a graph
    // leaf with its own stable storage (proven: materialised inputs are 0/N NaN
    // vs lazy 5/8). Autograd is taped on the PYTHON side (the Function's inputs),
    // so the kernel reading a data-identical settled copy does NOT detach grads.
    // Empty (0-elem, e.g. the rasterizer's placeholder colors/cov3D) stay as-is —
    // their data_ptr must remain null for the ext's `!= nullptr` dispatch.
    if (!borrowed.defined() || borrowed.numel() == 0) {
        stat_add_if(collect_stats, torch_ext_stats().input_empty_count);
        return borrowed;
    }
    stat_add_if(collect_stats, torch_ext_stats().input_clone_count);
    stat_add_if(collect_stats, torch_ext_stats().input_clone_bytes, nbytes);
    return borrowed.clone();   // graph-tracked settled copy, residency kept
}
Tensor tensor_from_pyvar_readonly(void* obj) {
    jittor::VarHolder* vh = jittor::from_py_object<jittor::VarHolder*>((PyObject*)obj);
    bool collect_stats = torch_ext_stats_enabled();
    stat_add_if(collect_stats, torch_ext_stats().direct_readonly_count);
    stat_add_if(collect_stats, torch_ext_stats().direct_readonly_bytes,
                collect_stats ? vh_nbytes_for_stats(vh) : 0);
    return adopt(vh, /*owns=*/false);
}
void* tensor_to_pyvar(const Tensor& t) {
    // Ext kernels launch on stream 0 through the c10 stream shim. Returning
    // without a global device barrier lets the next Jittor stream-0 op consume
    // this output in order. Do NOT bake here: baking output[i] runs jittor ops
    // mid-return that can disturb sibling outputs. Downstream stability is
    // handled by tensor_from_pyvar baking the way IN. Set
    // JITTOR_TORCH_EXT_SYNC_RETURN=1 or JITTOR_TORCH_EXT_SYNC_BOUNDARY=1 for
    // conservative debugging.
#ifdef HAS_CUDA
    if (torch_ext_sync_return_enabled())
        cudaDeviceSynchronize();
#endif
    bool collect_stats = torch_ext_stats_enabled();
    if (collect_stats && t.defined()) {
        stat_add_if(collect_stats, torch_ext_stats().output_count);
        stat_add_if(collect_stats, torch_ext_stats().output_bytes,
                    (long long)t.numel() * t.element_size());
    }
    return (void*)jittor::to_py_object<jittor::VarHolder*>(clone_holder(t._vh()));
}

} // namespace detail

// --------- Tensor methods ---------------------------------------------------
Tensor Tensor::view(IntArrayRef shape) const {
    static auto maker = jittor::get_op_info("reshape")
        .get_constructor<jittor::VarPtr, jittor::Var*, jittor::NanoVector>();
    jittor::VarPtr vp = maker(_vh()->var, to_nv(shape));
    return detail::adopt(new jittor::VarHolder(std::move(vp)), true);
}
Tensor Tensor::operator[](int64_t i) const {
    // row i along dim 0, as a copy (these exts index for reads/.item()).
    std::vector<int64_t> sh; detail::vh_shape(_vh(), sh);
    int64_t n0 = sh.empty() ? 1 : sh[0];
    if (i < 0) i += n0;
    std::vector<int64_t> sub(sh.begin() + (sh.empty() ? 0 : 1), sh.end());
    int64_t rest = 1; for (auto d : sub) rest *= d;
    int64_t esz = detail::vh_dsize(_vh());
    bool cuda = detail::vh_is_cuda(_vh());
    if (cuda && rest == 1 && torch_ext_native_scalar_index_enabled()) {
        stat_add(torch_ext_stats().native_scalar_index_count);
        static auto maker = jittor::get_op_info("getitem")
            .get_constructor<jittor::VarPtr, jittor::Var*, jittor::VarSlices&&>();
        jittor::VarSlices vs(1);
        vs.slices[0].set_int(i);
        jittor::VarPtr vp = maker(_vh()->var, std::move(vs));
        return detail::adopt(new jittor::VarHolder(std::move(vp)), true);
    }
    char* src = (char*)detail::vh_device_ptr(_vh()) + i * rest * esz;
    jittor::NanoVector nv = to_nv(IntArrayRef(sub));
    // Build a graph-tracked Var from the row (NOT empty()+cudaMemcpy(D2D), whose
    // result is NOT graph-settled, so a later .item()/sync re-runs the lazy empty
    // op and reads garbage — e.g. flex_gemm's `prefix_sum[-1].item<int32_t>()`
    // sizing a dynamic conv output returned INT_MIN). Stage through a host buffer
    // and migrate back to the device so residency is preserved (the bridge checks
    // is_cuda on tensors handed to triton kernels).
    int64_t nbytes = rest * esz;
    if (!cuda) return var_from_host_dev(src, nv, scalar_type(), false);
    stat_add(torch_ext_stats().host_scalar_index_count);
    stat_add(torch_ext_stats().host_scalar_index_bytes, nbytes);
    std::unique_ptr<char[]> host(new char[nbytes]);
    cudaMemcpy(host.get(), src, nbytes, cudaMemcpyDeviceToHost);
    return var_from_host_dev(host.get(), nv, scalar_type(), true);
}
Tensor Tensor::clone() const {
    bool cuda = detail::vh_is_cuda(_vh());
    jittor::NanoVector nv = to_nv(sizes());
    // Settled, graph-tracked clone (survives a later .item()/sync), residency
    // preserved — see operator[] for why a raw empty()+cudaMemcpy(D2D) is unsafe.
    if (cuda) return detail::adopt(make_copy_vh(_vh()->var), true);
    if (!cuda) return var_from_host_dev(detail::vh_device_ptr(_vh()), nv, scalar_type(), false);
}
Tensor Tensor::view(ScalarType st) const {
    int64_t old_sz = detail::vh_dsize(_vh());
    int64_t new_sz = scalar_size(st);
    int64_t bytes = numel() * old_sz;
    if (new_sz <= 0 || bytes % new_sz != 0)
        throw std::runtime_error("jtorch::Tensor::view(dtype) requires byte-compatible dtype");
    bool cuda = detail::vh_is_cuda(_vh());
    std::vector<int64_t> sh;
    detail::vh_shape(_vh(), sh);
    if (sh.empty()) {
        if (old_sz != new_sz)
            throw std::runtime_error("jtorch::Tensor::view(dtype) cannot change itemsize on a scalar");
    } else {
        int64_t last_bytes = sh.back() * old_sz;
        if (last_bytes % new_sz != 0)
            throw std::runtime_error("jtorch::Tensor::view(dtype) requires last dimension byte-compatible dtype");
        sh.back() = last_bytes / new_sz;
    }
    jittor::NanoVector nv = to_nv(sh);
    if (bytes == 0)
        return detail::adopt(make_var(nv, st, /*cpu=*/!cuda), true);
    if (torch_ext_dtype_view_native_enabled()) {
        stat_add(torch_ext_stats().dtype_view_native_count);
        stat_add(torch_ext_stats().dtype_view_native_bytes, bytes);
        return detail::adopt(make_reinterpret_view_vh(_vh()->var, nv, st), true);
    }
    if (!cuda)
        return var_from_host_dev(detail::vh_device_ptr(_vh()), nv, st, false);
    stat_add(torch_ext_stats().dtype_view_host_count);
    stat_add(torch_ext_stats().dtype_view_host_bytes, bytes);
    std::unique_ptr<char[]> host(new char[bytes]);
    cudaMemcpy(host.get(), detail::vh_device_ptr(_vh()), bytes, cudaMemcpyDeviceToHost);
    return var_from_host_dev(host.get(), nv, st, true);
}
Tensor& Tensor::resize_(IntArrayRef shape) {
    // in-place reallocation: swap the underlying Var for a fresh uninitialised one
    // of `shape`, preserving dtype + device residency. (diff-gaussian-rasterizer's
    // resizeFunctional starts buffers at {0} kByte then grows them to the exact
    // byte count the rasterizer needs, then reads .data_ptr().) New storage is a
    // plain empty op — no copy of the (discarded) old contents, matching torch's
    // resize_ which does not preserve data when the element count changes.
    bool cuda = detail::vh_is_cuda(_vh());
    *this = detail::adopt(make_var(to_nv(shape), scalar_type(), /*cpu=*/!cuda), true);
    return *this;
}

// --------- factories (device-aware: CUDA default, CPU honored) --------------
Tensor empty(IntArrayRef size, TensorOptions opt) {
    ScalarType st = opt.has_dtype_ ? opt.dtype_ : ScalarType::Float;
    return detail::adopt(make_var(to_nv(size), st, opt_is_cpu(opt)), true);
}
Tensor empty_like(const Tensor& t) {
    bool cpu = !detail::vh_is_cuda(t._vh());
    return detail::adopt(make_var(t._vh()->var->shape, t.scalar_type(), cpu), true);
}
Tensor empty_like(const Tensor& t, TensorOptions opt) {
    ScalarType st = opt.has_dtype_ ? opt.dtype_ : t.scalar_type();
    bool cpu = opt.has_device_ ? (opt.device_ == DeviceType::CPU) : !detail::vh_is_cuda(t._vh());
    return detail::adopt(make_var(t._vh()->var->shape, st, cpu), true);
}
Tensor zeros(IntArrayRef size, TensorOptions opt) {
    Tensor t = empty(size, opt);
    int64_t nbytes = t.numel() * detail::vh_dsize(t._vh());
    if (nbytes == 0)
        return t;
    void* p = t.data_ptr_void();
    if (detail::vh_allocator_is_cuda(t._vh())) cudaMemset(p, 0, nbytes);
    else std::memset(p, 0, nbytes);
    return t;
}
Tensor zeros_like(const Tensor& t) {
    bool cpu = !detail::vh_is_cuda(t._vh());
    return zeros(t.sizes(), cpu ? TensorOptions(t.scalar_type()).device(DeviceType::CPU)
                                : TensorOptions(t.scalar_type()).device(DeviceType::CUDA));
}
Tensor zeros_like(const Tensor& t, TensorOptions opt) {
    ScalarType st = opt.has_dtype_ ? opt.dtype_ : t.scalar_type();
    bool cpu = opt.has_device_ ? (opt.device_ == DeviceType::CPU) : !detail::vh_is_cuda(t._vh());
    return zeros(t.sizes(), TensorOptions(st).device(cpu ? DeviceType::CPU : DeviceType::CUDA));
}

template <typename T>
static Tensor _full_typed(IntArrayRef size, T value, TensorOptions opt, ScalarType st) {
    Tensor t = empty(size, TensorOptions(st).device(opt_is_cpu(opt)?DeviceType::CPU:DeviceType::CUDA));
    int64_t n = t.numel();
    if (n == 0)
        return t;
    void* p = t.data_ptr_void();
    if (detail::vh_allocator_is_cuda(t._vh())) {
        std::unique_ptr<T[]> host(new T[n]);
        for (int64_t i = 0; i < n; ++i) host[i] = value;
        cudaMemcpy(p, host.get(), n * sizeof(T), cudaMemcpyHostToDevice);
    } else {
        T* hp = (T*)p; for (int64_t i = 0; i < n; ++i) hp[i] = value;
    }
    return t;
}
static Tensor _full_memset(IntArrayRef size, TensorOptions opt, ScalarType st, int byte_value) {
    Tensor t = empty(size, TensorOptions(st).device(opt_is_cpu(opt)?DeviceType::CPU:DeviceType::CUDA));
    int64_t nbytes = t.numel() * detail::vh_dsize(t._vh());
    if (nbytes) {
        stat_add(torch_ext_stats().full_memset_count);
        stat_add(torch_ext_stats().full_memset_bytes, nbytes);
        if (opt_is_cpu(opt))
            std::memset(t.data_ptr_void(), byte_value, nbytes);
        else
            cudaMemset(t.data_ptr_void(), byte_value, nbytes);
    }
    return t;
}
static Tensor _full_bfloat16(IntArrayRef size, double value, TensorOptions opt) {
    Tensor t = empty(size, TensorOptions(ScalarType::BFloat16).device(opt_is_cpu(opt)?DeviceType::CPU:DeviceType::CUDA));
    int64_t n = t.numel();
    if (n == 0)
        return t;
    void* p = t.data_ptr_void();
    std::unique_ptr<__nv_bfloat16[]> host(new __nv_bfloat16[n]);
    __nv_bfloat16 v = __float2bfloat16((float)value);
    for (int64_t i = 0; i < n; ++i) host[i] = v;
    if (detail::vh_allocator_is_cuda(t._vh())) cudaMemcpy(p, host.get(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    else std::memcpy(p, host.get(), n * sizeof(__nv_bfloat16));
    return t;
}
// Saturating double->unsigned cast: the sentinel uint{32,64}_max the flex_gemm
// hashmap kernels pass through full()'s double `value` rounds above the type max,
// so clamp at/above-max to the exact max and floor negatives to 0.
template <typename U>
static inline U _sat_uint(double value) {
    if (!(value > 0.0)) return (U)0;
    double cap = (double)std::numeric_limits<U>::max();
    if (value >= cap) return std::numeric_limits<U>::max();
    return (U)value;
}
Tensor full(IntArrayRef size, double value, TensorOptions opt) {
    ScalarType st = opt.has_dtype_ ? opt.dtype_ : ScalarType::Float;
    if (value == 0.0 && !std::signbit(value))
        return zeros(size, TensorOptions(st).device(opt_is_cpu(opt)?DeviceType::CPU:DeviceType::CUDA));
    bool explicit_cuda = opt.has_device_ && opt.device_ == DeviceType::CUDA;
    if (explicit_cuda) {
        // Sparse kernels commonly initialize unsigned sentinel tensors to max().
        if (st == ScalarType::UInt16 && _sat_uint<uint16_t>(value) == std::numeric_limits<uint16_t>::max())
            return _full_memset(size, opt, st, 0xff);
        if (st == ScalarType::UInt32 && _sat_uint<uint32_t>(value) == std::numeric_limits<uint32_t>::max())
            return _full_memset(size, opt, st, 0xff);
        if (st == ScalarType::UInt64 && _sat_uint<uint64_t>(value) == std::numeric_limits<uint64_t>::max())
            return _full_memset(size, opt, st, 0xff);
    }
    switch (st) {
        case ScalarType::Float:  return _full_typed<float>(size, (float)value, opt, st);
        case ScalarType::Double: return _full_typed<double>(size, (double)value, opt, st);
        case ScalarType::Int:    return _full_typed<int32_t>(size, (int32_t)value, opt, st);
        case ScalarType::Long:   return _full_typed<int64_t>(size, (int64_t)value, opt, st);
        case ScalarType::Short:  return _full_typed<int16_t>(size, (int16_t)value, opt, st);
        case ScalarType::BFloat16:return _full_bfloat16(size, value, opt);
        case ScalarType::Byte:   return _full_typed<uint8_t>(size, (uint8_t)value, opt, st);
        case ScalarType::Bool:   return _full_typed<unsigned char>(size, (unsigned char)(value!=0), opt, st);
        // unsigned integer dtypes: flex_gemm's hashmap kernels do
        //   torch::full({...}, uint{32,64}_max, torch::dtype(torch::kUInt{16,32,64}))
        // to initialise a hashmap to its empty sentinel. `value` arrives here as a
        // double (the public full() signature), and uint{32,64}_max do NOT survive
        // the round-trip through double (2^64-1 rounds UP past the type max). Use a
        // SATURATING cast: any double at/above the type's max -> exact max (the
        // sentinel the kernels rely on for "empty slot"), otherwise the plain cast.
        case ScalarType::UInt16: return _full_typed<uint16_t>(size, _sat_uint<uint16_t>(value), opt, st);
        case ScalarType::UInt32: return _full_typed<uint32_t>(size, _sat_uint<uint32_t>(value), opt, st);
        case ScalarType::UInt64: return _full_typed<uint64_t>(size, _sat_uint<uint64_t>(value), opt, st);
        default: throw std::runtime_error("jtorch::full unsupported dtype");
    }
}
Tensor ones(IntArrayRef size, TensorOptions opt) { return full(size, 1.0, opt); }

Tensor from_blob(const void* data, IntArrayRef size, TensorOptions opt) {
    ScalarType st = opt.has_dtype_ ? opt.dtype_ : ScalarType::Float;
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, data);
    bool src_dev = (e == cudaSuccess && attr.type == cudaMemoryTypeDevice);
    if (e != cudaSuccess) cudaGetLastError();
    // target follows the source location unless an explicit device is requested
    bool tgt_cpu = opt.has_device_ ? (opt.device_ == DeviceType::CPU) : !src_dev;
    Tensor t = empty(size, TensorOptions(st).device(tgt_cpu?DeviceType::CPU:DeviceType::CUDA));
    int64_t nbytes = t.numel() * detail::vh_dsize(t._vh());
    void* dst = t.data_ptr_void();
    cudaMemcpyKind kind = src_dev ? (tgt_cpu ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice)
                                  : (tgt_cpu ? cudaMemcpyHostToHost   : cudaMemcpyHostToDevice);
    if (kind == cudaMemcpyHostToHost) std::memcpy(dst, data, nbytes);
    else cudaMemcpy(dst, data, nbytes, kind);
    return t;
}

// --------- ATen ops (wired on first real use) -------------------------------
// argsort / cumsum are needed by libtorch-style sparse extensions such as
// flex_gemm's neighbor-map post-processing. Prefer graph-tracked Jittor CUDA ops
// so large vectors stay on-device; keep the host implementations as conservative
// fallbacks for CPU tensors, missing optional CUDA ops, or unsupported shapes.

static bool torch_ext_native_aten_enabled() {
    if (env_truthy("JITTOR_TORCH_EXT_HOST_ATEN"))
        return false;
    return !env_falsey("JITTOR_TORCH_EXT_NATIVE_ATEN");
}

static Tensor _tensor_from_varptr(jittor::VarPtr&& vp) {
    return detail::adopt(new jittor::VarHolder(std::move(vp)), true);
}

static Tensor _argsort_jittor_op(const Tensor& self, int64_t dim, bool descending) {
    static auto maker = jittor::get_op_info("argsort")
        .get_constructor<std::vector<jittor::VarPtr>, jittor::Var*, int, bool, jittor::NanoString>();
    std::vector<jittor::VarPtr> outs = maker(self._vh()->var, (int)dim, descending,
                                             to_ns(ScalarType::Long));
    if (outs.empty())
        throw std::runtime_error("jtorch::argsort native op returned no outputs");
    return _tensor_from_varptr(std::move(outs[0]));
}

static bool _cumsum_dim_supported_by_cub(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    if (nd == 1)
        return dim == 0 || dim == -1;
    if (nd == 2)
        return dim == 1 || dim == -1;
    return false;
}

static Tensor _cumsum_jittor_op(const Tensor& self, ScalarType out_st) {
    if (!jittor::has_op("cub_cumsum"))
        throw std::runtime_error("jtorch::cumsum native cub_cumsum op is unavailable");
    static auto make_unary = jittor::get_op_info("unary")
        .get_constructor<jittor::VarPtr, jittor::Var*, jittor::NanoString>();
    static auto make_cub_cumsum = jittor::get_op_info("cub_cumsum")
        .get_constructor<jittor::VarPtr, jittor::Var*, bool>();

    jittor::Var* src = self._vh()->var;
    jittor::VarPtr casted;
    jittor::NanoString out_ns = to_ns(out_st);
    if (src->dtype() != out_ns) {
        casted = make_unary(src, out_ns);
        src = casted;
    }
    jittor::VarPtr out = make_cub_cumsum(src, false);
    return _tensor_from_varptr(std::move(out));
}

template <typename K>
static Tensor _argsort_1d_typed(const Tensor& self, bool descending) {
    int64_t n = self.numel();
    const K* kp = self.data_ptr<K>();
    bool cuda = detail::vh_is_cuda(self._vh());
    // do the argsort on the HOST (operands are small index vectors). Synchronise
    // first so any prior write (jittor stream OR a raw <<<>>> kernel) is visible.
    if (cuda) cudaDeviceSynchronize();
    std::unique_ptr<K[]> hk(new K[n]);
    if (cuda) {
        stat_add(torch_ext_stats().host_fallback_copy_count);
        stat_add(torch_ext_stats().host_fallback_copy_bytes, n * (long long)sizeof(K));
        cudaMemcpy(hk.get(), kp, n * sizeof(K), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(hk.get(), kp, n * sizeof(K));
    }
    std::vector<int64_t> v(n); for (int64_t i = 0; i < n; ++i) v[i] = i;
    std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) {
        return descending ? (hk[a] > hk[b]) : (hk[a] < hk[b]); });
    // build the result as a graph-tracked Var from the host indices, resident on
    // the same device as the input (a CUDA consumer — e.g. flex_gemm's triton
    // kernel reading `sorted_idx` — needs a device pointer, not host memory).
    return var_from_host_dev(v.data(), self._vh()->var->shape, ScalarType::Long, cuda);
}

Tensor argsort(const Tensor& self, int64_t dim, bool descending) {
    int64_t nd = self.dim();
    if (dim < 0) dim += nd;
    if (nd > 1 && dim != nd - 1)
        throw std::runtime_error("jtorch::argsort only supports last-dim 1-D sort");
    bool cuda = detail::vh_is_cuda(self._vh());
    if (cuda && torch_ext_native_aten_enabled()) {
        switch (self.scalar_type()) {
            case ScalarType::Int:
            case ScalarType::Long:
            case ScalarType::Float:
            case ScalarType::UInt32:
            case ScalarType::UInt64:
                return _argsort_jittor_op(self, dim, descending);
            default:
                break;
        }
    }
    switch (self.scalar_type()) {
        case ScalarType::Int:    return _argsort_1d_typed<int32_t>(self, descending);
        case ScalarType::Long:   return _argsort_1d_typed<int64_t>(self, descending);
        case ScalarType::Float:  return _argsort_1d_typed<float>(self, descending);
        case ScalarType::UInt32: return _argsort_1d_typed<uint32_t>(self, descending);
        case ScalarType::UInt64: return _argsort_1d_typed<uint64_t>(self, descending);
        default: throw std::runtime_error("jtorch::argsort unsupported dtype");
    }
}

// inclusive prefix-sum, done on the HOST (the operands here are tiny segment
// vectors). Robust against the caller's data living on jittor's stream or a raw
// <<<>>> kernel's default stream — we cudaDeviceSynchronize() first so every
// prior write (jittor-tracked or not) is visible before we read.
template <typename SRC, typename DST>
static std::unique_ptr<DST[]> _cumsum_host(const SRC* sp, int64_t n, bool cuda) {
    if (cuda) cudaDeviceSynchronize();
    std::unique_ptr<SRC[]> hi(new SRC[n]);
    if (cuda) {
        stat_add(torch_ext_stats().host_fallback_copy_count);
        stat_add(torch_ext_stats().host_fallback_copy_bytes, n * (long long)sizeof(SRC));
        cudaMemcpy(hi.get(), sp, n * sizeof(SRC), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(hi.get(), sp, n * sizeof(SRC));
    }
    std::unique_ptr<DST[]> ho(new DST[n]);
    DST acc = 0;
    for (int64_t i = 0; i < n; ++i) { acc += (DST)hi[i]; ho[i] = acc; }
    return ho;
}

Tensor cumsum(const Tensor& self, int64_t dim, ScalarType out_st) {
    int64_t nd = self.dim();
    if (dim < 0) dim += nd;
    if (nd > 1 && !(dim == 0 || dim == nd - 1))
        throw std::runtime_error("jtorch::cumsum only supports a 1-D / contiguous scan");
    bool cuda = detail::vh_is_cuda(self._vh());
    int64_t n = self.numel();
    if (cuda && torch_ext_native_aten_enabled() && _cumsum_dim_supported_by_cub(self, dim)) {
        switch (self.scalar_type()) {
            case ScalarType::Int:
            case ScalarType::Long:
            case ScalarType::Bool:
            case ScalarType::Byte:
                if (out_st == ScalarType::Int || out_st == ScalarType::Long)
                    return _cumsum_jittor_op(self, out_st);
                break;
            default:
                break;
        }
    }
    // prefix-sum on the host, then build a graph-tracked Var from the result so a
    // later .item()/.numel()/sync reads the actual values (not a re-run empty op).
#define _JT_CUMSUM_BUILD(SRC) do { \
        const SRC* sp = self.data_ptr<SRC>(); \
        if (out_st == ScalarType::Int) { \
            auto ho = _cumsum_host<SRC,int32_t>(sp, n, cuda); \
            return var_from_host_dev(ho.get(), self._vh()->var->shape, ScalarType::Int, cuda); \
        } else if (out_st == ScalarType::Long) { \
            auto ho = _cumsum_host<SRC,int64_t>(sp, n, cuda); \
            return var_from_host_dev(ho.get(), self._vh()->var->shape, ScalarType::Long, cuda); \
        } else throw std::runtime_error("jtorch::cumsum unsupported out dtype"); \
    } while(0)
    switch (self.scalar_type()) {
        case ScalarType::Int:  _JT_CUMSUM_BUILD(int32_t); break;
        case ScalarType::Long: _JT_CUMSUM_BUILD(int64_t); break;
        case ScalarType::Bool: _JT_CUMSUM_BUILD(unsigned char); break;
        case ScalarType::Byte: _JT_CUMSUM_BUILD(uint8_t); break;
        default: throw std::runtime_error("jtorch::cumsum unsupported in dtype");
    }
#undef _JT_CUMSUM_BUILD
    throw std::runtime_error("jtorch::cumsum unreachable");
}
Tensor cumsum(const Tensor& self, int64_t dim) { return cumsum(self, dim, self.scalar_type()); }

template <typename T>
static std::unique_ptr<T[]> _copy_to_host_typed(const Tensor& self) {
    int64_t n = self.numel();
    std::unique_ptr<T[]> h(new T[n]);
    if (n == 0) return h;
    bool cuda = detail::vh_is_cuda(self._vh());
    if (cuda) {
        stat_add(torch_ext_stats().host_fallback_copy_count);
        stat_add(torch_ext_stats().host_fallback_copy_bytes, n * (long long)sizeof(T));
        cudaDeviceSynchronize();
        cudaMemcpy(h.get(), self.data_ptr<T>(), n * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(h.get(), self.data_ptr<T>(), n * sizeof(T));
    }
    return h;
}

template <typename T>
static Tensor _tensor_from_host_same_device(const T* data, const Tensor& like,
                                            const jittor::NanoVector& shape,
                                            ScalarType st) {
    return var_from_host_dev(data, shape, st, detail::vh_is_cuda(like._vh()));
}

template <typename T>
static std::tuple<Tensor, Tensor> _sort_1d_typed(const Tensor& self, bool descending) {
    int64_t n = self.numel();
    auto h = _copy_to_host_typed<T>(self);
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return descending ? (h[a] > h[b]) : (h[a] < h[b]);
    });
    std::unique_ptr<T[]> vals(new T[n]);
    for (int64_t i = 0; i < n; ++i) vals[i] = h[idx[i]];
    jittor::NanoVector sh = self._vh()->var->shape;
    Tensor out_vals = _tensor_from_host_same_device(vals.get(), self, sh, self.scalar_type());
    Tensor out_idx = _tensor_from_host_same_device(idx.data(), self, sh, ScalarType::Long);
    return std::make_tuple(out_vals, out_idx);
}

std::tuple<Tensor, Tensor> sort(const Tensor& self, int64_t dim, bool descending) {
    if (self.dim() > 1 && !(dim == -1 || dim == self.dim() - 1))
        throw std::runtime_error("jtorch::sort only supports last-dim 1-D sort");
    switch (self.scalar_type()) {
        case ScalarType::Byte:   return _sort_1d_typed<uint8_t>(self, descending);
        case ScalarType::Char:   return _sort_1d_typed<int8_t>(self, descending);
        case ScalarType::Short:  return _sort_1d_typed<int16_t>(self, descending);
        case ScalarType::Int:    return _sort_1d_typed<int32_t>(self, descending);
        case ScalarType::Long:   return _sort_1d_typed<int64_t>(self, descending);
        case ScalarType::Float:  return _sort_1d_typed<float>(self, descending);
        case ScalarType::Double: return _sort_1d_typed<double>(self, descending);
        case ScalarType::UInt16: return _sort_1d_typed<uint16_t>(self, descending);
        case ScalarType::UInt32: return _sort_1d_typed<uint32_t>(self, descending);
        case ScalarType::UInt64: return _sort_1d_typed<uint64_t>(self, descending);
        default: throw std::runtime_error("jtorch::sort unsupported dtype");
    }
}

template <typename T, typename IDX>
static Tensor _index_select_typed(const Tensor& self, int64_t dim, const Tensor& index) {
    std::vector<int64_t> in_shape;
    detail::vh_shape(self._vh(), in_shape);
    int64_t nd = (int64_t)in_shape.size();
    if (nd == 0)
        throw std::runtime_error("jtorch::index_select cannot index a scalar");
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("jtorch::index_select dim out of range");
    int64_t nidx = index.numel();
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= in_shape[i];
    for (int64_t i = dim + 1; i < nd; ++i) inner *= in_shape[i];
    int64_t dim_size = in_shape[dim];
    auto hsrc = _copy_to_host_typed<T>(self);
    auto hidx = _copy_to_host_typed<IDX>(index);
    int64_t out_numel = outer * nidx * inner;
    std::unique_ptr<T[]> out(new T[out_numel]);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < nidx; ++i) {
            int64_t j = (int64_t)hidx[i];
            if (j < 0) j += dim_size;
            if (j < 0 || j >= dim_size)
                throw std::runtime_error("jtorch::index_select index out of range");
            const T* src = hsrc.get() + (o * dim_size + j) * inner;
            T* dst = out.get() + (o * nidx + i) * inner;
            std::memcpy(dst, src, inner * sizeof(T));
        }
    }
    in_shape[dim] = nidx;
    return _tensor_from_host_same_device(out.get(), self, to_nv(in_shape), self.scalar_type());
}

template <typename T>
static Tensor _index_select_dispatch_index(const Tensor& self, int64_t dim, const Tensor& index) {
    switch (index.scalar_type()) {
        case ScalarType::Int:  return _index_select_typed<T, int32_t>(self, dim, index);
        case ScalarType::Long: return _index_select_typed<T, int64_t>(self, dim, index);
        default: throw std::runtime_error("jtorch::index_select index must be int32/int64");
    }
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t nd = self.dim();
    if (dim < 0) dim += nd;
    switch (self.scalar_type()) {
        case ScalarType::Byte:   return _index_select_dispatch_index<uint8_t>(self, dim, index);
        case ScalarType::Char:   return _index_select_dispatch_index<int8_t>(self, dim, index);
        case ScalarType::Short:  return _index_select_dispatch_index<int16_t>(self, dim, index);
        case ScalarType::Int:    return _index_select_dispatch_index<int32_t>(self, dim, index);
        case ScalarType::Long:   return _index_select_dispatch_index<int64_t>(self, dim, index);
        case ScalarType::Float:  return _index_select_dispatch_index<float>(self, dim, index);
        case ScalarType::Double: return _index_select_dispatch_index<double>(self, dim, index);
        case ScalarType::UInt16: return _index_select_dispatch_index<uint16_t>(self, dim, index);
        case ScalarType::UInt32: return _index_select_dispatch_index<uint32_t>(self, dim, index);
        case ScalarType::UInt64: return _index_select_dispatch_index<uint64_t>(self, dim, index);
        default: throw std::runtime_error("jtorch::index_select unsupported dtype");
    }
}

template <typename T>
static std::tuple<Tensor, Tensor> _unique_typed(const Tensor& self, bool sorted, bool return_inverse) {
    int64_t n = self.numel();
    auto h = _copy_to_host_typed<T>(self);
    std::vector<T> vals;
    vals.reserve(n);
    std::vector<int64_t> inv(n, 0);
    std::map<T, int64_t> seen;
    for (int64_t i = 0; i < n; ++i) {
        auto it = seen.find(h[i]);
        if (it == seen.end()) {
            int64_t pos = (int64_t)vals.size();
            seen[h[i]] = pos;
            vals.push_back(h[i]);
            inv[i] = pos;
        } else {
            inv[i] = it->second;
        }
    }
    if (sorted) {
        std::vector<T> sorted_vals = vals;
        std::sort(sorted_vals.begin(), sorted_vals.end());
        if (return_inverse) {
            std::map<T, int64_t> new_pos;
            for (int64_t i = 0; i < (int64_t)sorted_vals.size(); ++i)
                new_pos[sorted_vals[i]] = i;
            for (int64_t i = 0; i < n; ++i)
                inv[i] = new_pos[h[i]];
        }
        vals.swap(sorted_vals);
    }
    jittor::NanoVector vsh;
    vsh.push_back((int64_t)vals.size());
    jittor::NanoVector ish;
    ish.push_back(n);
    Tensor unique = _tensor_from_host_same_device(vals.data(), self, vsh, self.scalar_type());
    Tensor inverse = _tensor_from_host_same_device(inv.data(), self, ish, ScalarType::Long);
    return std::make_tuple(unique, inverse);
}

std::tuple<Tensor, Tensor> _unique(const Tensor& self, bool sorted, bool return_inverse) {
    switch (self.scalar_type()) {
        case ScalarType::Byte:   return _unique_typed<uint8_t>(self, sorted, return_inverse);
        case ScalarType::Char:   return _unique_typed<int8_t>(self, sorted, return_inverse);
        case ScalarType::Short:  return _unique_typed<int16_t>(self, sorted, return_inverse);
        case ScalarType::Int:    return _unique_typed<int32_t>(self, sorted, return_inverse);
        case ScalarType::Long:   return _unique_typed<int64_t>(self, sorted, return_inverse);
        case ScalarType::Float:  return _unique_typed<float>(self, sorted, return_inverse);
        case ScalarType::Double: return _unique_typed<double>(self, sorted, return_inverse);
        case ScalarType::UInt16: return _unique_typed<uint16_t>(self, sorted, return_inverse);
        case ScalarType::UInt32: return _unique_typed<uint32_t>(self, sorted, return_inverse);
        case ScalarType::UInt64: return _unique_typed<uint64_t>(self, sorted, return_inverse);
        default: throw std::runtime_error("jtorch::_unique unsupported dtype");
    }
}

template <typename T>
static Tensor _masked_select_typed(const Tensor& self, const Tensor& mask) {
    if (self.numel() != mask.numel())
        throw std::runtime_error("jtorch::masked_select requires same numel mask");
    auto h = _copy_to_host_typed<T>(self);
    auto m = _copy_to_host_typed<unsigned char>(mask);
    std::vector<T> out;
    out.reserve(self.numel());
    for (int64_t i = 0; i < self.numel(); ++i)
        if (m[i]) out.push_back(h[i]);
    jittor::NanoVector sh;
    sh.push_back((int64_t)out.size());
    return _tensor_from_host_same_device(out.data(), self, sh, self.scalar_type());
}

template <typename T>
static Tensor _ne_tensor_typed(const Tensor& self, const Tensor& other) {
    if (self.numel() != other.numel())
        throw std::runtime_error("jtorch::ne requires equal numel tensors");
    auto a = _copy_to_host_typed<T>(self);
    auto b = _copy_to_host_typed<T>(other);
    std::unique_ptr<unsigned char[]> out(new unsigned char[self.numel()]);
    for (int64_t i = 0; i < self.numel(); ++i)
        out[i] = (unsigned char)(a[i] != b[i]);
    return _tensor_from_host_same_device(out.get(), self, self._vh()->var->shape, ScalarType::Bool);
}

template <typename T>
static Tensor _ne_scalar_typed(const Tensor& self, double other) {
    auto a = _copy_to_host_typed<T>(self);
    T v = (T)other;
    std::unique_ptr<unsigned char[]> out(new unsigned char[self.numel()]);
    for (int64_t i = 0; i < self.numel(); ++i)
        out[i] = (unsigned char)(a[i] != v);
    return _tensor_from_host_same_device(out.get(), self, self._vh()->var->shape, ScalarType::Bool);
}

// --------- Tensor method-form ops (wired on first real use) -----------------
Tensor Tensor::cumsum(int64_t dim) const { return jtorch::cumsum(*this, dim); }
std::tuple<Tensor, Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return jtorch::sort(*this, dim, descending);
}
Tensor Tensor::masked_select(const Tensor& mask) const {
    if (mask.scalar_type() != ScalarType::Bool && mask.scalar_type() != ScalarType::Byte)
        throw std::runtime_error("jtorch::masked_select mask must be bool/uint8");
    switch (scalar_type()) {
        case ScalarType::Byte:   return _masked_select_typed<uint8_t>(*this, mask);
        case ScalarType::Char:   return _masked_select_typed<int8_t>(*this, mask);
        case ScalarType::Short:  return _masked_select_typed<int16_t>(*this, mask);
        case ScalarType::Int:    return _masked_select_typed<int32_t>(*this, mask);
        case ScalarType::Long:   return _masked_select_typed<int64_t>(*this, mask);
        case ScalarType::Float:  return _masked_select_typed<float>(*this, mask);
        case ScalarType::Double: return _masked_select_typed<double>(*this, mask);
        case ScalarType::UInt16: return _masked_select_typed<uint16_t>(*this, mask);
        case ScalarType::UInt32: return _masked_select_typed<uint32_t>(*this, mask);
        case ScalarType::UInt64: return _masked_select_typed<uint64_t>(*this, mask);
        default: throw std::runtime_error("jtorch::masked_select unsupported dtype");
    }
}
Tensor Tensor::ne(const Tensor& other) const {
    if (scalar_type() != other.scalar_type())
        throw std::runtime_error("jtorch::ne currently requires equal dtype tensors");
    switch (scalar_type()) {
        case ScalarType::Byte:   return _ne_tensor_typed<uint8_t>(*this, other);
        case ScalarType::Char:   return _ne_tensor_typed<int8_t>(*this, other);
        case ScalarType::Short:  return _ne_tensor_typed<int16_t>(*this, other);
        case ScalarType::Int:    return _ne_tensor_typed<int32_t>(*this, other);
        case ScalarType::Long:   return _ne_tensor_typed<int64_t>(*this, other);
        case ScalarType::Float:  return _ne_tensor_typed<float>(*this, other);
        case ScalarType::Double: return _ne_tensor_typed<double>(*this, other);
        case ScalarType::UInt16: return _ne_tensor_typed<uint16_t>(*this, other);
        case ScalarType::UInt32: return _ne_tensor_typed<uint32_t>(*this, other);
        case ScalarType::UInt64: return _ne_tensor_typed<uint64_t>(*this, other);
        default: throw std::runtime_error("jtorch::ne unsupported dtype");
    }
}
Tensor Tensor::ne_scalar(double other) const {
    switch (scalar_type()) {
        case ScalarType::Byte:   return _ne_scalar_typed<uint8_t>(*this, other);
        case ScalarType::Char:   return _ne_scalar_typed<int8_t>(*this, other);
        case ScalarType::Short:  return _ne_scalar_typed<int16_t>(*this, other);
        case ScalarType::Int:    return _ne_scalar_typed<int32_t>(*this, other);
        case ScalarType::Long:   return _ne_scalar_typed<int64_t>(*this, other);
        case ScalarType::Float:  return _ne_scalar_typed<float>(*this, other);
        case ScalarType::Double: return _ne_scalar_typed<double>(*this, other);
        case ScalarType::UInt16: return _ne_scalar_typed<uint16_t>(*this, other);
        case ScalarType::UInt32: return _ne_scalar_typed<uint32_t>(*this, other);
        case ScalarType::UInt64: return _ne_scalar_typed<uint64_t>(*this, other);
        default: throw std::runtime_error("jtorch::ne_scalar unsupported dtype");
    }
}

} // namespace jtorch
