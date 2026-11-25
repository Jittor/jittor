#include "ops/composite/cuda_launch_op.h"
#include "core/executor.h"
#include "core/var.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "runtime/tensor_placement.h"
#ifdef HAS_CUDA
#include <mutex>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#endif

namespace jittor {
namespace {

#ifdef HAS_CUDA
using CuLaunchKernelFn = int (*)(
    void*, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned,
    void*, void**, void**);
using CuCtxGetCurrentFn = int (*)(void**);
using CuCtxSetCurrentFn = int (*)(void*);
using CuGetErrorStringFn = int (*)(int, const char**);

struct CudaDriverSymbols {
    CuLaunchKernelFn launch_kernel = nullptr;
    CuCtxGetCurrentFn ctx_get_current = nullptr;
    CuCtxSetCurrentFn ctx_set_current = nullptr;
    CuGetErrorStringFn get_error_string = nullptr;
};

static const CudaDriverSymbols& cuda_driver_symbols() {
    static CudaDriverSymbols syms;
    static std::once_flag once;
    std::call_once(once, []() {
#ifndef _WIN32
        void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        if (!lib) lib = RTLD_DEFAULT;
        syms.launch_kernel = reinterpret_cast<CuLaunchKernelFn>(dlsym(lib, "cuLaunchKernel"));
        syms.ctx_get_current = reinterpret_cast<CuCtxGetCurrentFn>(dlsym(lib, "cuCtxGetCurrent"));
        syms.ctx_set_current = reinterpret_cast<CuCtxSetCurrentFn>(dlsym(lib, "cuCtxSetCurrent"));
        syms.get_error_string = reinterpret_cast<CuGetErrorStringFn>(dlsym(lib, "cuGetErrorString"));
#endif
    });
    return syms;
}
#endif

static inline bool is_power_of_two_or_zero(uint64 x) {
    return (x & (x - 1)) == 0;
}

} // namespace

void CudaLaunchOp::parse_and_validate(const vector<uint64>& launch_config,
                                      const vector<int>& extra_meta) {
    USER_CHECK(launch_config.size() == 15)
        << "cuda_launch: launch_config must contain 15 elements, got " << launch_config.size();
    function_handle = launch_config[0];
    launcher_handle = launch_config[1];
    ctx_handle = launch_config[2];
    gx = static_cast<uint>(launch_config[3]);
    gy = static_cast<uint>(launch_config[4]);
    gz = static_cast<uint>(launch_config[5]);
    bx = static_cast<uint>(launch_config[6]);
    by = static_cast<uint>(launch_config[7]);
    bz = static_cast<uint>(launch_config[8]);
    shared_mem = static_cast<uint>(launch_config[9]);
    scratch_bytes = launch_config[10];
    scratch_align = launch_config[11];
    profile_scratch_bytes = launch_config[12];
    profile_scratch_align = launch_config[13];
    expected_device = static_cast<int>(launch_config[14]);

    USER_CHECK(function_handle != 0) << "cuda_launch: null CUDA function handle";
    USER_CHECK(bx > 0 && by > 0 && bz > 0) << "cuda_launch: block dimensions must be positive";
    USER_CHECK(expected_device >= 0) << "cuda_launch: invalid expected CUDA device " << expected_device;
    USER_CHECK(is_power_of_two_or_zero(scratch_align)
            && is_power_of_two_or_zero(profile_scratch_align))
        << "cuda_launch: scratch alignment must be a power of two";
    USER_CHECK(param_kinds.size() == param_values.size())
        << "cuda_launch: param_kinds and param_values size mismatch";

    for (size_t i = 0; i < param_kinds.size(); ++i) {
        int kind = param_kinds[i];
        uint64 val = param_values[i];
        if (kind == kInputPtr) {
            USER_CHECK(val < read_buffers.size())
                << "cuda_launch: input slot index out of range at param " << i;
        } else if (kind == kOutputPtr) {
            USER_CHECK(val < write_buffers.size())
                << "cuda_launch: output slot index out of range at param " << i;
        } else if (kind < kInputPtr || kind > kScalar64) {
            USER_ERROR << "cuda_launch: unknown parameter kind " << kind;
        }
    }

    if (!extra_meta.empty()) {
        size_t pos = 0;
        USER_CHECK(pos < extra_meta.size()) << "cuda_launch: malformed extra_meta";
        int n_aligned = extra_meta[pos++];
        USER_CHECK(n_aligned >= 0 && pos + (size_t)n_aligned <= extra_meta.size())
            << "cuda_launch: malformed aligned_slots in extra_meta";
        aligned_slots.reserve(n_aligned);
        for (int i = 0; i < n_aligned; ++i) {
            int slot = extra_meta[pos++];
            USER_CHECK(slot >= 0 && (size_t)slot < param_kinds.size())
                << "cuda_launch: aligned slot out of range";
            aligned_slots.push_back(slot);
        }

        if (pos < extra_meta.size()) {
            int n_copies = extra_meta[pos++];
            USER_CHECK(n_copies >= 0 && pos + (size_t)n_copies * 2 <= extra_meta.size())
                << "cuda_launch: malformed copy pairs in extra_meta";
            copy_dst_outs.reserve(n_copies);
            copy_src_ins.reserve(n_copies);
            for (int i = 0; i < n_copies; ++i) {
                int dst = extra_meta[pos++];
                int src = extra_meta[pos++];
                USER_CHECK(dst >= 0 && (size_t)dst < write_buffers.size()
                        && src >= 0 && (size_t)src < read_buffers.size())
                    << "cuda_launch: copy pair index out of range";
                copy_dst_outs.push_back(dst);
                copy_src_ins.push_back(src);
            }
        }

        if (pos < extra_meta.size()) {
            int n_bounds = extra_meta[pos++];
            USER_CHECK(n_bounds >= 0 && pos + (size_t)n_bounds * 2 <= extra_meta.size())
                << "cuda_launch: malformed bound pairs in extra_meta";
            bound_out_indices.reserve(n_bounds);
            bound_in_indices.reserve(n_bounds);
            for (int i = 0; i < n_bounds; ++i) {
                int out_i = extra_meta[pos++];
                int in_i = extra_meta[pos++];
                USER_CHECK(out_i >= 0 && (size_t)out_i < write_buffers.size()
                        && in_i >= 0 && (size_t)in_i < read_buffers.size())
                    << "cuda_launch: bound pair index out of range";
                bound_out_indices.push_back(out_i);
                bound_in_indices.push_back(in_i);
            }
        }
        USER_CHECK(pos == extra_meta.size()) << "cuda_launch: trailing bytes in extra_meta";
    }
}

CudaLaunchOp::CudaLaunchOp(
        vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes,
        vector<Var*>&& inputs,
        vector<uint64>&& launch_config,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& extra_meta)
    : read_buffers(inputs),
      param_kinds(move(param_kinds)),
      param_values(move(param_values)) {
    USER_CHECK(!shapes.empty() && shapes.size() == dtypes.size())
        << "cuda_launch needs matching nonempty output shapes and dtypes";
    USER_CHECK(accelerator_backend_id() == BackendId::Cuda)
        << "cuda_launch requires CUDA backend support";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);

    const int target_dev = launch_config.size() >= 15 ? static_cast<int>(launch_config[14]) : 0;
    TensorPlacement zero_input_placement;
    if (read_buffers.empty()) {
        USER_CHECK(target_dev >= 0) << "cuda_launch: invalid CUDA device " << target_dev;
        zero_input_placement = TensorPlacement({BackendId::Cuda, target_dev});
    }
    write_buffers.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int j = 0; j < shapes[i].size(); ++j)
            USER_CHECK(shapes[i][j] >= 0) << "cuda_launch requires concrete output shapes";
        Var* out = create_output(shapes[i], dtypes[i]);
        if (read_buffers.empty()) {
            out->placement = zero_input_placement;
            out->device_id = target_dev;
        }
        write_buffers.push_back(out);
    }
    parse_and_validate(launch_config, extra_meta);
}

CudaLaunchOp::CudaLaunchOp(
        vector<Var*>&& inputs,
        vector<Var*>&& outputs,
        vector<uint64>&& launch_config,
        vector<int>&& param_kinds,
        vector<uint64>&& param_values,
        vector<int>&& extra_meta)
    : read_buffers(inputs),
      param_kinds(move(param_kinds)),
      param_values(move(param_values)) {
    USER_CHECK(!outputs.empty()) << "cuda_launch requires at least one output";
    USER_CHECK(accelerator_backend_id() == BackendId::Cuda)
        << "cuda_launch requires CUDA backend support";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);

    write_buffers.reserve(outputs.size());
    for (Var* o : outputs) {
        Var* out = create_output(o->shape, o->dtype());
        out->share_with(o);
        write_buffers.push_back(out);
    }
    parse_and_validate(launch_config, extra_meta);
    order_bound_outputs = !bound_out_indices.empty();
}

void CudaLaunchOp::infer_shape() {
    if (!order_bound_outputs) return;
    order_bound_outputs = false;
    vector<Var*> written;
    written.reserve(write_buffers.size());
    for (Var* output : write_buffers)
        if (output->share_src) written.push_back(output->share_src);
    if (!written.empty())
        order_after_readers(this, written);
}

void CudaLaunchOp::run() {
#ifndef HAS_CUDA
    USER_ERROR << "cuda_launch requires CUDA backend support";
#else
    USER_CHECK(!write_buffers.empty()) << "cuda_launch requires at least one output";
    Device device = allocation_device(write_buffers[0]->allocator);
    USER_CHECK(device.backend == BackendId::Cuda && device.index == expected_device)
        << "cuda_launch: expected CUDA device " << expected_device
        << ", got backend " << int(device.backend) << " device " << device.index;
    for (Var* v : read_buffers) {
        Device d = allocation_device(v->allocator);
        USER_CHECK(d.backend == BackendId::Cuda && d.index == expected_device)
            << "cuda_launch: all input buffers must be on CUDA device " << expected_device;
    }
    for (Var* v : write_buffers) {
        Device d = allocation_device(v->allocator);
        USER_CHECK(d.backend == BackendId::Cuda && d.index == expected_device)
            << "cuda_launch: all output buffers must be on CUDA device " << expected_device;
    }

    if (!bound_out_indices.empty()) {
        for (size_t i = 0; i < bound_out_indices.size(); ++i) {
            Var* out_v = write_buffers[bound_out_indices[i]];
            Var* in_v = read_buffers[bound_in_indices[i]];
            USER_CHECK(out_v->mem_ptr == in_v->mem_ptr)
                << "prepared Triton output lost its storage binding";
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
                        << "prepared Triton bound outputs overlap";
                }
            }
        }
    }

    BackendStream stream = backend_stream(device, BackendStreamKind::Compute);
    for (size_t i = 0; i < copy_dst_outs.size(); ++i) {
        Var* dst = write_buffers[copy_dst_outs[i]];
        Var* src = read_buffers[copy_src_ins[i]];
        if (dst->size)
            backend_copy_async(dst->mem_ptr, device, src->mem_ptr, device, dst->size, stream);
    }

    if (gx == 0 || gy == 0 || gz == 0) return;

    const auto& syms = cuda_driver_symbols();
    if (ctx_handle && syms.ctx_get_current && syms.ctx_set_current) {
        void* current_ctx = nullptr;
        if (syms.ctx_get_current(&current_ctx) != 0
                || current_ctx != reinterpret_cast<void*>(ctx_handle)) {
            int rc = syms.ctx_set_current(reinterpret_cast<void*>(ctx_handle));
            USER_CHECK(rc == 0) << "cuda_launch: cuCtxSetCurrent failed with CUresult " << rc;
        }
    }

    const size_t s_align = scratch_align > 1 ? static_cast<size_t>(scratch_align) : 1;
    Allocation scratch_alloc = scratch_bytes > 0
        ? runtime_executor().alloc_temp(static_cast<size_t>(scratch_bytes) + (s_align > 1 ? s_align : 0))
        : Allocation();
    void* scratch_ptr = scratch_alloc.ptr
        ? reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(scratch_alloc.ptr) + s_align - 1) & ~(s_align - 1))
        : nullptr;
    const size_t p_align = profile_scratch_align > 1 ? static_cast<size_t>(profile_scratch_align) : 1;
    Allocation profile_scratch_alloc = profile_scratch_bytes > 0
        ? runtime_executor().alloc_temp(static_cast<size_t>(profile_scratch_bytes) + (p_align > 1 ? p_align : 0))
        : Allocation();
    void* profile_scratch_ptr = profile_scratch_alloc.ptr
        ? reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(profile_scratch_alloc.ptr) + p_align - 1) & ~(p_align - 1))
        : nullptr;

    constexpr size_t kStackParams = 64;
    const size_t nparams = param_kinds.size();
    uint64 stack_storage[kStackParams];
    void* stack_ptrs[kStackParams];
    vector<uint64> heap_storage;
    vector<void*> heap_ptrs;
    uint64* storage = stack_storage;
    void** ptrs = stack_ptrs;
    if (nparams > kStackParams) {
        heap_storage.resize(nparams);
        heap_ptrs.resize(nparams);
        storage = heap_storage.data();
        ptrs = heap_ptrs.data();
    }

    for (size_t i = 0; i < nparams; ++i) {
        switch (param_kinds[i]) {
            case kInputPtr:
                storage[i] = reinterpret_cast<uint64>(read_buffers[param_values[i]]->mem_ptr);
                break;
            case kOutputPtr:
                storage[i] = reinterpret_cast<uint64>(write_buffers[param_values[i]]->mem_ptr);
                break;
            case kScratchPtr:
                storage[i] = reinterpret_cast<uint64>(scratch_ptr);
                break;
            case kProfileScratchPtr:
                storage[i] = reinterpret_cast<uint64>(profile_scratch_ptr);
                break;
            case kNullPtr:
                storage[i] = 0;
                break;
            case kScalar32:
            case kScalar64:
                storage[i] = param_values[i];
                break;
            default:
                USER_ERROR << "cuda_launch: invalid parameter kind " << param_kinds[i];
        }
        ptrs[i] = &storage[i];
    }

    for (int slot : aligned_slots) {
        USER_CHECK((storage[slot] & 15u) == 0)
            << "unaligned prepared Triton pointer at slot " << slot;
    }

    CuLaunchKernelFn launcher = launcher_handle
        ? reinterpret_cast<CuLaunchKernelFn>(launcher_handle)
        : syms.launch_kernel;
    USER_CHECK(launcher != nullptr) << "cuda_launch: cuLaunchKernel symbol is unavailable";
    const int status = launcher(
        reinterpret_cast<void*>(function_handle),
        gx, gy, gz, bx, by, bz, shared_mem,
        stream.handle, ptrs, nullptr);
    if (PREDICT_BRANCH_NOT_TAKEN(status != 0)) {
        const char* err_str = nullptr;
        if (syms.get_error_string) syms.get_error_string(status, &err_str);
        USER_ERROR << "cuda_launch: cuLaunchKernel failed with CUresult " << status
                   << " (" << (err_str ? err_str : "unknown error") << ")";
    }
#endif
}

VarPtr CudaLaunchOp::grad(Var*, Var*, Var*, int) {
    LOGf << "cuda_launch has no implicit derivative; define backward with jittor.Function";
    return nullptr;
}

} // namespace jittor
