// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "codegen/nvrtc_compiler.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

#include "bindings/pyjt/gil.h"
#include "core/fused_op.h"
#include "core/op.h"
#include "core/var.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/array_op.h"
#include "ops/op_register.h"
#include "ops/ternary_op.h"
#include "ops/unary_op.h"
#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
#include "runtime/jit_policy.h"
#include "runtime/profiler/profiler.h"
#include "runtime/profiler/step_trace.h"
#include "utils/cache_compile.h"
#include "utils/flags.h"

namespace jittor {

DEFINE_FLAG(int, use_nvrtc, 1,
    "Compile eligible CUDA fused elementwise kernels in-process with NVRTC "
    "instead of invoking nvcc (1=enabled, 0=disabled).");

DECLARE_FLAG(string, nvcc_path);
DECLARE_FLAG(string, cache_path);
DECLARE_FLAG(vector<int>, cuda_archs);
DECLARE_FLAG(int, profiler_enable);
DECLARE_FLAG(int, jit_search_kernel);

namespace {

std::atomic<int64> g_nvrtc_compile_count{0};
std::atomic<int64> g_nvrtc_disk_hit_count{0};
std::atomic<int64> g_nvrtc_launch_count{0};
std::atomic<int64> g_nvrtc_fallback_count{0};

constexpr uint32_t kNvrtcCacheMagic = 0x4A544E56u;  // "JTNV"
constexpr uint32_t kNvrtcCacheVersion = 1u;

#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)

// Minimal NVRTC & CUDA Driver C ABI declarations (zero build-time header/link dep).
using nvrtcResult = int;
using nvrtcProgram = void*;
constexpr nvrtcResult NVRTC_SUCCESS = 0;

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUcontext = void*;
using CUdevice = int;
constexpr CUresult CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct NvrtcApi {
    bool initialized = false;
    bool available = false;
    void* builtins_handle = nullptr;
    void* nvrtc_handle = nullptr;
    void* cuda_handle = nullptr;

    nvrtcResult (*nvrtcVersion)(int*, int*) = nullptr;
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*,
                                      int, const char* const*, const char* const*) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;

    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, void*, void**) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**) = nullptr;
    CUresult (*cuCtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*cuCtxGetDevice)(CUdevice*) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
};

static NvrtcApi& nvrtc_api() {
    static NvrtcApi* api = new NvrtcApi();
    static std::once_flag once;
    std::call_once(once, [&]() {
        api->initialized = true;

        // 1. Load libcuda driver API
        api->cuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!api->cuda_handle) api->cuda_handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
        if (!api->cuda_handle) {
            LOGv << "NVRTC disabled: cannot dlopen libcuda.so.1";
            return;
        }
        auto load_cu = [&](auto& fn, const char* sym) -> bool {
            using FnT = typename std::remove_reference<decltype(fn)>::type;
            fn = reinterpret_cast<FnT>(dlsym(api->cuda_handle, sym));
            return fn != nullptr;
        };
        if (!load_cu(api->cuModuleLoadDataEx, "cuModuleLoadDataEx") ||
            !load_cu(api->cuModuleGetFunction, "cuModuleGetFunction") ||
            !load_cu(api->cuLaunchKernel, "cuLaunchKernel") ||
            !load_cu(api->cuCtxGetCurrent, "cuCtxGetCurrent") ||
            !load_cu(api->cuCtxGetDevice, "cuCtxGetDevice") ||
            !load_cu(api->cuDeviceGetAttribute, "cuDeviceGetAttribute")) {
            LOGv << "NVRTC disabled: missing required CUDA driver symbols";
            return;
        }
        load_cu(api->cuGetErrorString, "cuGetErrorString");

        // 2. Collect candidate directories for libnvrtc.so and libnvrtc-builtins.so
        std::vector<std::string> lib_dirs;
        auto add_dir = [&](const std::string& d) {
            if (d.empty()) return;
            if (std::find(lib_dirs.begin(), lib_dirs.end(), d) == lib_dirs.end())
                lib_dirs.push_back(d);
        };
        auto add_cuda_root = [&](const std::string& root) {
            if (root.empty()) return;
            add_dir(root + "/lib64");
            add_dir(root + "/lib");
            add_dir(root + "/targets/x86_64-linux/lib");
            add_dir(root + "/targets/aarch64-linux/lib");
        };
        if (!nvcc_path.empty()) {
            // nvcc_path is typically <cuda_root>/bin/nvcc
            auto slash = nvcc_path.rfind('/');
            if (slash != std::string::npos) {
                std::string bin_dir = nvcc_path.substr(0, slash);
                auto slash2 = bin_dir.rfind('/');
                if (slash2 != std::string::npos) {
                    std::string cuda_root = bin_dir.substr(0, slash2);
                    add_cuda_root(cuda_root);
                    // Pip CUDA 12 wheel layout: .../nvidia/cuda_nvcc/bin/nvcc
                    // has sibling .../nvidia/cuda_nvrtc/lib/libnvrtc.so.12
                    auto slash3 = cuda_root.rfind('/');
                    if (slash3 != std::string::npos) {
                        add_dir(cuda_root.substr(0, slash3) + "/cuda_nvrtc/lib");
                    }
                }
            }
        }
        if (const char* ch = std::getenv("CUDA_HOME")) add_cuda_root(ch);
        if (const char* ch = std::getenv("CUDA_PATH")) add_cuda_root(ch);
        add_cuda_root("/usr/local/cuda");

        auto try_preload_builtins = [&](const std::string& dir) {
            if (api->builtins_handle) return;
            const char* direct[] = {
                "libnvrtc-builtins.so",
                "libnvrtc-builtins.so.13.0",
                "libnvrtc-builtins.so.12.8",
                "libnvrtc-builtins.so.12.6",
                "libnvrtc-builtins.so.12.4",
                "libnvrtc-builtins.so.12.2",
                "libnvrtc-builtins.so.12.1",
                "libnvrtc-builtins.so.12.0",
            };
            for (const char* name : direct) {
                std::string p = dir + "/" + name;
                if (access(p.c_str(), R_OK) == 0) {
                    void* h = dlopen(p.c_str(), RTLD_LAZY | RTLD_GLOBAL);
                    if (h) { api->builtins_handle = h; return; }
                }
            }
            DIR* dp = opendir(dir.c_str());
            if (!dp) return;
            while (struct dirent* ent = readdir(dp)) {
                std::string n = ent->d_name;
                if (n.rfind("libnvrtc-builtins.so", 0) == 0 && n.find(".alt.") == std::string::npos) {
                    std::string p = dir + "/" + n;
                    void* h = dlopen(p.c_str(), RTLD_LAZY | RTLD_GLOBAL);
                    if (h) { api->builtins_handle = h; break; }
                }
            }
            closedir(dp);
        };

        auto try_open_nvrtc_in_dir = [&](const std::string& dir) -> bool {
            const char* names[] = {
                "libnvrtc.so",
                "libnvrtc.so.13",
                "libnvrtc.so.12",
                "libnvrtc.so.11.2",
            };
            for (const char* name : names) {
                std::string p = dir + "/" + name;
                if (access(p.c_str(), R_OK) == 0) {
                    try_preload_builtins(dir);
                    void* h = dlopen(p.c_str(), RTLD_LAZY | RTLD_LOCAL);
                    if (h) {
                        api->nvrtc_handle = h;
                        LOGv << "Loaded NVRTC from" << p;
                        return true;
                    }
                }
            }
            return false;
        };

        for (const auto& dir : lib_dirs) {
            if (try_open_nvrtc_in_dir(dir)) break;
        }
        if (!api->nvrtc_handle) {
            const char* sys_names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"};
            for (const char* n : sys_names) {
                void* h = dlopen(n, RTLD_LAZY | RTLD_LOCAL);
                if (h) { api->nvrtc_handle = h; break; }
            }
        }
        void* lookup_handle = api->nvrtc_handle ? api->nvrtc_handle : RTLD_DEFAULT;
        if (!api->nvrtc_handle && dlsym(RTLD_DEFAULT, "nvrtcCreateProgram") == nullptr) {
            LOGv << "NVRTC disabled: cannot dlopen libnvrtc.so";
            return;
        }

        auto load_nvrtc = [&](auto& fn, const char* sym) -> bool {
            using FnT = typename std::remove_reference<decltype(fn)>::type;
            fn = reinterpret_cast<FnT>(dlsym(lookup_handle, sym));
            return fn != nullptr;
        };
        if (!load_nvrtc(api->nvrtcVersion, "nvrtcVersion") ||
            !load_nvrtc(api->nvrtcCreateProgram, "nvrtcCreateProgram") ||
            !load_nvrtc(api->nvrtcDestroyProgram, "nvrtcDestroyProgram") ||
            !load_nvrtc(api->nvrtcCompileProgram, "nvrtcCompileProgram") ||
            !load_nvrtc(api->nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize") ||
            !load_nvrtc(api->nvrtcGetProgramLog, "nvrtcGetProgramLog") ||
            !load_nvrtc(api->nvrtcGetPTXSize, "nvrtcGetPTXSize") ||
            !load_nvrtc(api->nvrtcGetPTX, "nvrtcGetPTX")) {
            LOGv << "NVRTC disabled: missing required libnvrtc symbols";
            return;
        }
        // Optional CUBIN API (CUDA 11.1+)
        load_nvrtc(api->nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
        load_nvrtc(api->nvrtcGetCUBIN, "nvrtcGetCUBIN");

        int maj = 0, min = 0;
        if (api->nvrtcVersion(&maj, &min) == NVRTC_SUCCESS) {
            LOGv << "NVRTC initialized: version" << maj >> "." >> min;
        }
        api->available = true;
    });
    return *api;
}

static NvrtcDType to_nvrtc_dtype(NanoString ns) {
    if (ns == ns_float32) return NvrtcDType::F32;
    if (ns == ns_float64) return NvrtcDType::F64;
    if (ns == ns_float16) return NvrtcDType::F16;
    if (ns == ns_bfloat16) return NvrtcDType::BF16;
    if (ns == ns_int32) return NvrtcDType::I32;
    if (ns == ns_int64) return NvrtcDType::I64;
    if (ns == ns_int8) return NvrtcDType::I8;
    if (ns == ns_uint8) return NvrtcDType::U8;
    if (ns == ns_bool) return NvrtcDType::BOOL;
    return NvrtcDType::INVALID;
}

static int nvrtc_dtype_size(NvrtcDType dt) {
    switch (dt) {
        case NvrtcDType::F32:
        case NvrtcDType::I32: return 4;
        case NvrtcDType::F64:
        case NvrtcDType::I64: return 8;
        case NvrtcDType::F16:
        case NvrtcDType::BF16: return 2;
        case NvrtcDType::I8:
        case NvrtcDType::U8:
        case NvrtcDType::BOOL: return 1;
        default: return 0;
    }
}

static bool is_float_dtype(NvrtcDType dt) {
    return dt == NvrtcDType::F32 || dt == NvrtcDType::F64 ||
           dt == NvrtcDType::F16 || dt == NvrtcDType::BF16;
}

static bool is_half_dtype(NvrtcDType dt) {
    return dt == NvrtcDType::F16 || dt == NvrtcDType::BF16;
}

static bool is_signed_int_dtype(NvrtcDType dt) {
    return dt == NvrtcDType::I8 || dt == NvrtcDType::I32 || dt == NvrtcDType::I64;
}

static const char* storage_ctype(NvrtcDType dt) {
    switch (dt) {
        case NvrtcDType::F32: return "float";
        case NvrtcDType::F64: return "double";
        case NvrtcDType::F16: return "unsigned short";
        case NvrtcDType::BF16: return "unsigned short";
        case NvrtcDType::I32: return "int";
        case NvrtcDType::I64: return "long long";
        case NvrtcDType::I8: return "signed char";
        case NvrtcDType::U8: return "unsigned char";
        case NvrtcDType::BOOL: return "unsigned char";
        default: return "void";
    }
}

static const char* vec4_ctype(NvrtcDType dt) {
    switch (dt) {
        case NvrtcDType::F32: return "float4";
        case NvrtcDType::I32: return "int4";
        case NvrtcDType::F16:
        case NvrtcDType::BF16: return "ushort4";
        case NvrtcDType::I8: return "char4";
        case NvrtcDType::U8:
        case NvrtcDType::BOOL: return "uchar4";
        default: return nullptr;
    }
}

static bool unary_op_supported(NanoString op, NvrtcDType tx, NvrtcDType ty) {
    if (tx == NvrtcDType::INVALID || ty == NvrtcDType::INVALID) return false;
    if (op == ns_cast || op == ns_conj || op == ns_logical_not) return true;
    if (op == ns_bitwise_not) return !is_float_dtype(tx) && !is_float_dtype(ty);
    if (op == ns_abs) return tx == ty || (is_float_dtype(tx) && is_float_dtype(ty));
    if (op == ns_negative) return tx != NvrtcDType::BOOL && ty != NvrtcDType::BOOL;
    if (op == ns_relu) return true;
    if (op == ns_round_int || op == ns_floor_int || op == ns_ceil_int)
        return is_float_dtype(tx) && (ty == NvrtcDType::I32 || ty == NvrtcDType::I64);
    if (op == ns_exp || op == ns_log || op == ns_log1p || op == ns_sqrt ||
        op == ns_round || op == ns_floor || op == ns_ceil ||
        op == ns_sin || op == ns_asin || op == ns_sinh || op == ns_asinh ||
        op == ns_cos || op == ns_acos || op == ns_cosh || op == ns_acosh ||
        op == ns_tan || op == ns_atan || op == ns_tanh || op == ns_atanh ||
        op == ns_sigmoid || op == ns_erf || op == ns_erfinv) {
        return is_float_dtype(ty);
    }
    return false;
}

static bool binary_op_supported(NanoString op, NvrtcDType tx, NvrtcDType ty, NvrtcDType tz) {
    if (tx == NvrtcDType::INVALID || ty == NvrtcDType::INVALID || tz == NvrtcDType::INVALID)
        return false;
    if (op == ns_divide && !is_float_dtype(tz)) return false;
    if (tz == NvrtcDType::BOOL) {
        return op == ns_less || op == ns_less_equal ||
               op == ns_greater || op == ns_greater_equal ||
               op == ns_equal || op == ns_not_equal ||
               op == ns_logical_and || op == ns_logical_or || op == ns_logical_xor ||
               op == ns_bitwise_and || op == ns_bitwise_or || op == ns_bitwise_xor ||
               op == ns_add || op == ns_subtract || op == ns_multiply ||
               op == ns_maximum || op == ns_minimum;
    }
    if (op == ns_add || op == ns_subtract || op == ns_multiply || op == ns_divide ||
        op == ns_floor_divide || op == ns_mod || op == ns_fmod || op == ns_pow ||
        op == ns_maximum || op == ns_minimum ||
        op == ns_less || op == ns_less_equal || op == ns_greater || op == ns_greater_equal ||
        op == ns_equal || op == ns_not_equal ||
        op == ns_logical_and || op == ns_logical_or || op == ns_logical_xor)
        return true;
    if (op == ns_bitwise_and || op == ns_bitwise_or || op == ns_bitwise_xor ||
        op == ns_left_shift || op == ns_right_shift) {
        return !is_float_dtype(tx) && !is_float_dtype(ty) && !is_float_dtype(tz);
    }
    return false;
}

// Convert expression of dtype `src` to `dst` in storage C type.
static std::string emit_cast(const std::string& expr, NvrtcDType src, NvrtcDType dst) {
    if (src == dst) return expr;
    // Step 1: decode half/bf16 to float if needed
    std::string fval = expr;
    if (src == NvrtcDType::F16) fval = "_jt_f16_to_f32(" + expr + ")";
    else if (src == NvrtcDType::BF16) fval = "_jt_bf16_to_f32(" + expr + ")";

    // Step 2: encode to dst
    if (dst == NvrtcDType::F16) return "_jt_f32_to_f16((float)(" + fval + "))";
    if (dst == NvrtcDType::BF16) return "_jt_f32_to_bf16((float)(" + fval + "))";
    if (dst == NvrtcDType::BOOL) return "(unsigned char)((" + fval + ") != 0)";
    return "((" + std::string(storage_ctype(dst)) + ")(" + fval + "))";
}

static std::string emit_as_float(const std::string& expr, NvrtcDType src) {
    if (src == NvrtcDType::F16) return "_jt_f16_to_f32(" + expr + ")";
    if (src == NvrtcDType::BF16) return "_jt_bf16_to_f32(" + expr + ")";
    if (src == NvrtcDType::F32) return expr;
    return "((float)(" + expr + "))";
}

static std::string emit_as_double(const std::string& expr, NvrtcDType src) {
    if (src == NvrtcDType::F16) return "((double)_jt_f16_to_f32(" + expr + "))";
    if (src == NvrtcDType::BF16) return "((double)_jt_bf16_to_f32(" + expr + "))";
    if (src == NvrtcDType::F64) return expr;
    return "((double)(" + expr + "))";
}

static std::string emit_as_bool(const std::string& expr, NvrtcDType src) {
    if (src == NvrtcDType::F16) return "(_jt_f16_to_f32(" + expr + ") != 0.0f)";
    if (src == NvrtcDType::BF16) return "(_jt_bf16_to_f32(" + expr + ") != 0.0f)";
    return "((" + expr + ") != 0)";
}

static std::string emit_from_float(const std::string& expr, NvrtcDType dst) {
    if (dst == NvrtcDType::F32) return expr;
    if (dst == NvrtcDType::F16) return "_jt_f32_to_f16(" + expr + ")";
    if (dst == NvrtcDType::BF16) return "_jt_f32_to_bf16(" + expr + ")";
    if (dst == NvrtcDType::BOOL) return "(unsigned char)((" + expr + ") != 0.0f)";
    return "((" + std::string(storage_ctype(dst)) + ")(" + expr + "))";
}

static std::string emit_unary_expr(NanoString op, const std::string& x, NvrtcDType tx, NvrtcDType ty) {
    if (op == ns_cast) return emit_cast(x, tx, ty);
    if (op == ns_conj) return emit_cast(x, tx, ty);
    if (op == ns_logical_not) return "(unsigned char)(!" + emit_as_bool(x, tx) + ")";
    if (op == ns_bitwise_not) {
        if (ty == NvrtcDType::BOOL) return "(unsigned char)(!" + emit_as_bool(x, tx) + ")";
        return "((" + std::string(storage_ctype(ty)) + ")(~(" + emit_cast(x, tx, ty) + ")))";
    }
    if (op == ns_negative) {
        if (is_half_dtype(ty)) return emit_from_float("(-" + emit_as_float(x, tx) + ")", ty);
        return "((" + std::string(storage_ctype(ty)) + ")(-" + emit_cast(x, tx, ty) + "))";
    }
    if (op == ns_abs) {
        if (ty == NvrtcDType::F32) return "fabsf(" + emit_as_float(x, tx) + ")";
        if (ty == NvrtcDType::F64) return "fabs(" + emit_as_double(x, tx) + ")";
        if (is_half_dtype(ty)) return emit_from_float("fabsf(" + emit_as_float(x, tx) + ")", ty);
        if (is_signed_int_dtype(ty)) {
            std::string vx = emit_cast(x, tx, ty);
            if (ty == NvrtcDType::I64)
                return "((" + vx + ") < 0 ? (long long)(0ull - (unsigned long long)(" + vx + ")) : (" + vx + "))";
            return "((" + vx + ") < 0 ? (" + std::string(storage_ctype(ty)) + ")(0u - (unsigned int)(" + vx + ")) : (" + vx + "))";
        }
        return emit_cast(x, tx, ty);
    }
    if (op == ns_relu) {
        if (is_half_dtype(ty)) {
            std::string f = emit_as_float(x, tx);
            return emit_from_float("((" + f + ") > 0.0f ? (" + f + ") : 0.0f)", ty);
        }
        std::string vx = emit_cast(x, tx, ty);
        return "((" + vx + ") > 0 ? (" + vx + ") : (" + std::string(storage_ctype(ty)) + ")0)";
    }
    if (op == ns_round_int) return "((" + std::string(storage_ctype(ty)) + ")roundf(" + emit_as_float(x, tx) + "))";
    if (op == ns_floor_int) return "((" + std::string(storage_ctype(ty)) + ")floorf(" + emit_as_float(x, tx) + "))";
    if (op == ns_ceil_int) return "((" + std::string(storage_ctype(ty)) + ")ceilf(" + emit_as_float(x, tx) + "))";

    if (op == ns_sigmoid) {
        if (ty == NvrtcDType::F64) {
            std::string d = emit_as_double(x, tx);
            return "(1.0 / (1.0 + exp(-(" + d + "))))";
        }
        std::string f = emit_as_float(x, tx);
        if (ty == NvrtcDType::F16) {
            return "_jt_f32_to_f16(1.0f / (1.0f + expf(fminf(-(" + f + "), 30.0f))))";
        }
        if (ty == NvrtcDType::BF16) {
            return "_jt_f32_to_bf16(1.0f / (1.0f + expf(fminf(-(" + f + "), 300.0f))))";
        }
        return "(1.0f / (1.0f + expf(-(" + f + "))))";
    }

    auto math_fn = [&](const char* f32_fn, const char* f64_fn) -> std::string {
        if (ty == NvrtcDType::F64) return std::string(f64_fn) + "(" + emit_as_double(x, tx) + ")";
        std::string res = std::string(f32_fn) + "(" + emit_as_float(x, tx) + ")";
        return emit_from_float(res, ty);
    };

    if (op == ns_exp) return math_fn("expf", "exp");
    if (op == ns_log) return math_fn("logf", "log");
    if (op == ns_log1p) return math_fn("log1pf", "log1p");
    if (op == ns_sqrt) return math_fn("sqrtf", "sqrt");
    if (op == ns_round) return math_fn("rintf", "rint");
    if (op == ns_floor) return math_fn("floorf", "floor");
    if (op == ns_ceil) return math_fn("ceilf", "ceil");
    if (op == ns_sin) return math_fn("sinf", "sin");
    if (op == ns_asin) return math_fn("asinf", "asin");
    if (op == ns_sinh) return math_fn("sinhf", "sinh");
    if (op == ns_asinh) return math_fn("asinhf", "asinh");
    if (op == ns_cos) return math_fn("cosf", "cos");
    if (op == ns_acos) return math_fn("acosf", "acos");
    if (op == ns_cosh) return math_fn("coshf", "cosh");
    if (op == ns_acosh) return math_fn("acoshf", "acosh");
    if (op == ns_tan) return math_fn("tanf", "tan");
    if (op == ns_atan) return math_fn("atanf", "atan");
    if (op == ns_tanh) return math_fn("tanhf", "tanh");
    if (op == ns_atanh) return math_fn("atanhf", "atanh");
    if (op == ns_erf) return math_fn("erff", "erf");
    if (op == ns_erfinv) return math_fn("erfinvf", "erfinv");
    return emit_cast(x, tx, ty);
}

static std::string emit_binary_expr(
    NanoString op,
    const std::string& x, NvrtcDType tx,
    const std::string& y, NvrtcDType ty,
    NvrtcDType tz
) {
    if (op == ns_logical_and)
        return "(unsigned char)(" + emit_as_bool(x, tx) + " && " + emit_as_bool(y, ty) + ")";
    if (op == ns_logical_or)
        return "(unsigned char)(" + emit_as_bool(x, tx) + " || " + emit_as_bool(y, ty) + ")";
    if (op == ns_logical_xor)
        return "(unsigned char)(" + emit_as_bool(x, tx) + " != " + emit_as_bool(y, ty) + ")";

    // Comparisons
    auto emit_cmp = [&](const char* cmp) -> std::string {
        std::string a, b;
        if (tx == NvrtcDType::F64 || ty == NvrtcDType::F64) {
            a = emit_as_double(x, tx);
            b = emit_as_double(y, ty);
        } else if (is_float_dtype(tx) || is_float_dtype(ty)) {
            a = emit_as_float(x, tx);
            b = emit_as_float(y, ty);
        } else if (tx == ty) {
            a = x;
            b = y;
        } else {
            a = "((long long)(" + x + "))";
            b = "((long long)(" + y + "))";
        }
        return "(unsigned char)((" + a + ") " + cmp + " (" + b + "))";
    };
    if (op == ns_less) return emit_cmp("<");
    if (op == ns_less_equal) return emit_cmp("<=");
    if (op == ns_greater) return emit_cmp(">");
    if (op == ns_greater_equal) return emit_cmp(">=");
    if (op == ns_equal) return emit_cmp("==");
    if (op == ns_not_equal) return emit_cmp("!=");

    // Arithmetic ops (+, -, *, /)
    if (tz == NvrtcDType::BOOL) {
        if (op == ns_add || op == ns_maximum)
            return "(unsigned char)(" + emit_as_bool(x, tx) + " || " + emit_as_bool(y, ty) + ")";
        if (op == ns_multiply || op == ns_minimum)
            return "(unsigned char)(" + emit_as_bool(x, tx) + " && " + emit_as_bool(y, ty) + ")";
        if (op == ns_subtract)
            return "(unsigned char)(" + emit_as_bool(x, tx) + " != " + emit_as_bool(y, ty) + ")";
    }
    auto emit_arith = [&](const char* sym) -> std::string {
        if (is_half_dtype(tz)) {
            std::string a = emit_as_float(x, tx);
            std::string b = emit_as_float(y, ty);
            return emit_from_float("((" + a + ") " + sym + " (" + b + "))", tz);
        }
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        return "((" + std::string(storage_ctype(tz)) + ")((" + a + ") " + sym + " (" + b + ")))";
    };
    if (op == ns_add) return emit_arith("+");
    if (op == ns_subtract) return emit_arith("-");
    if (op == ns_multiply) return emit_arith("*");
    if (op == ns_divide) return emit_arith("/");

    if (op == ns_maximum || op == ns_minimum) {
        const char* fn = (op == ns_maximum) ? "_jt_max" : "_jt_min";
        if (is_half_dtype(tz)) {
            return emit_from_float(
                std::string(fn) + "_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")", tz);
        }
        if (tz == NvrtcDType::F32)
            return std::string(fn) + "_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")";
        if (tz == NvrtcDType::F64)
            return std::string(fn) + "_f64(" + emit_as_double(x, tx) + ", " + emit_as_double(y, ty) + ")";
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        const char* cmp = (op == ns_maximum) ? ">" : "<";
        return "((" + a + ") " + cmp + " (" + b + ") ? (" + a + ") : (" + b + "))";
    }

    if (op == ns_pow) {
        if (is_half_dtype(tz)) {
            return emit_from_float(
                "_jt_signed_pow_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")", tz);
        }
        // Match common_op_type_cuda_map: jittor::_signed_pow(double, double) cast to tz
        const std::string base = emit_as_double(x, tx);
        const std::string exponent = emit_as_double(y, ty);
        const std::string power = "_jt_signed_pow_f64(" + base + ", " + exponent + ")";
        if (tz == NvrtcDType::F32 && tx == NvrtcDType::F32) {
            const std::string f32 = emit_as_float(x, tx);
            return "((" + exponent + " == 2.0) ? __fmul_rn(" + f32 + ", " + f32
                + ") : (float)((" + exponent + " == 3.0) ? " + base + " * " + base
                + " * " + base + " : " + power + "))";
        }
        if (tz == NvrtcDType::F64) return power;
        return "((" + std::string(storage_ctype(tz)) + ")(" + power + "))";
    }

    if (op == ns_floor_divide) {
        if (is_half_dtype(tz))
            return emit_from_float("_jt_floor_div_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")", tz);
        if (tz == NvrtcDType::F32)
            return "_jt_floor_div_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")";
        if (tz == NvrtcDType::F64)
            return "_jt_floor_div_f64(" + emit_as_double(x, tx) + ", " + emit_as_double(y, ty) + ")";
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        if (tz == NvrtcDType::I32) return "_jt_floor_div_i32(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I64) return "_jt_floor_div_i64(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I8) return "(signed char)_jt_floor_div_i32((int)(" + a + "), (int)(" + b + "))";
        return "((" + b + ") == 0 ? (" + std::string(storage_ctype(tz)) + ")0 : (" + std::string(storage_ctype(tz)) + ")((" + a + ") / (" + b + ")))";
    }

    if (op == ns_mod) {
        if (is_half_dtype(tz))
            return emit_from_float("_jt_floor_mod_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")", tz);
        if (tz == NvrtcDType::F32)
            return "_jt_floor_mod_f32(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")";
        if (tz == NvrtcDType::F64)
            return "_jt_floor_mod_f64(" + emit_as_double(x, tx) + ", " + emit_as_double(y, ty) + ")";
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        if (tz == NvrtcDType::I32) return "_jt_floor_mod_i32(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I64) return "_jt_floor_mod_i64(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I8) return "(signed char)_jt_floor_mod_i32((int)(" + a + "), (int)(" + b + "))";
        return "((" + b + ") == 0 ? (" + std::string(storage_ctype(tz)) + ")0 : (" + std::string(storage_ctype(tz)) + ")((" + a + ") % (" + b + ")))";
    }

    if (op == ns_fmod) {
        if (is_half_dtype(tz))
            return emit_from_float("fmodf(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")", tz);
        if (tz == NvrtcDType::F32)
            return "fmodf(" + emit_as_float(x, tx) + ", " + emit_as_float(y, ty) + ")";
        if (tz == NvrtcDType::F64)
            return "fmod(" + emit_as_double(x, tx) + ", " + emit_as_double(y, ty) + ")";
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        if (tz == NvrtcDType::I32) return "_jt_fmod_i32(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I64) return "_jt_fmod_i64(" + a + ", " + b + ")";
        if (tz == NvrtcDType::I8) return "(signed char)_jt_fmod_i32((int)(" + a + "), (int)(" + b + "))";
        return "((" + b + ") == 0 ? (" + std::string(storage_ctype(tz)) + ")0 : (" + std::string(storage_ctype(tz)) + ")((" + a + ") % (" + b + ")))";
    }

    // Bitwise & shift
    auto emit_bit = [&](const char* sym) -> std::string {
        if (tz == NvrtcDType::BOOL) {
            return "(unsigned char)((" + emit_as_bool(x, tx) + ") " + sym + " (" + emit_as_bool(y, ty) + "))";
        }
        std::string a = emit_cast(x, tx, tz);
        std::string b = emit_cast(y, ty, tz);
        return "((" + std::string(storage_ctype(tz)) + ")((" + a + ") " + sym + " (" + b + ")))";
    };
    if (op == ns_bitwise_and) return emit_bit("&");
    if (op == ns_bitwise_or) return emit_bit("|");
    if (op == ns_bitwise_xor) return emit_bit("^");
    if (op == ns_left_shift) return emit_bit("<<");
    if (op == ns_right_shift) return emit_bit(">>");

    return emit_cast(x, tx, tz);
}

static int detect_sm_arch() {
    auto& api = nvrtc_api();
    if (api.available) {
        CUcontext ctx = nullptr;
        CUdevice dev = 0;
        if (api.cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx &&
            api.cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
            int maj = 0, min = 0;
            if (api.cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == CUDA_SUCCESS &&
                api.cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == CUDA_SUCCESS &&
                maj > 0) {
                return maj * 10 + min;
            }
        }
    }
    if (!cuda_archs.empty()) {
        int dev = current_device();
        if (dev >= 0 && (size_t)dev < cuda_archs.size() && cuda_archs[dev] > 0)
            return cuda_archs[dev];
        if (cuda_archs[0] > 0) return cuda_archs[0];
    }
    return 0;
}

static std::string extract_math_policy(const std::string& jit_key) {
    for (const auto& kv : parse_jit_keys(jit_key)) {
        if (kv.first == "JIT_cuda_math") return kv.second;
    }
    return runtime_flag_cuda_kernel_math();
}

static const char* kNvrtcPrelude = R"CUDA(
__device__ __forceinline__ float _jt_f16_to_f32(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short _jt_f32_to_f16(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float _jt_bf16_to_f32(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    return __uint_as_float(u);
}
__device__ __forceinline__ unsigned short _jt_f32_to_bf16(float f) {
#if __CUDA_ARCH__ >= 800
    unsigned short b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
#else
    unsigned int bits = __float_as_uint(f);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    unsigned int lsb = (bits >> 16) & 1u;
    return (unsigned short)((bits + 0x7fffu + lsb) >> 16);
#endif
}
__device__ __forceinline__ float _jt_max_f32(float a, float b) {
    return ((a > b) | (a != a)) ? a : b;
}
__device__ __forceinline__ float _jt_min_f32(float a, float b) {
    return ((a < b) | (a != a)) ? a : b;
}
__device__ __forceinline__ double _jt_max_f64(double a, double b) {
    return ((a > b) | (a != a)) ? a : b;
}
__device__ __forceinline__ double _jt_min_f64(double a, double b) {
    return ((a < b) | (a != a)) ? a : b;
}
__device__ __forceinline__ float _jt_signed_pow_f32(float x, float y) {
    if (x < 0.0f && floorf(y) == y)
        return powf(-x, y) * (fmodf(y, 2.0f) != 0.0f ? -1.0f : 1.0f);
    return powf(x, y);
}
__device__ __forceinline__ double _jt_signed_pow_f64(double x, double y) {
    if (x < 0.0 && floor(y) == y)
        return pow(-x, y) * (fmod(y, 2.0) != 0.0 ? -1.0 : 1.0);
    return pow(x, y);
}
__device__ __forceinline__ float _jt_floor_div_f32(float x, float y) {
    if (y == 0.0f) return x / y;
    float mod = fmodf(x, y);
    float div = (x - mod) / y;
    if (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) div -= 1.0f;
    if (div == 0.0f) return copysignf(0.0f, x / y);
    float fd = floorf(div);
    if (div - fd > 0.5f) fd += 1.0f;
    return fd;
}
__device__ __forceinline__ double _jt_floor_div_f64(double x, double y) {
    if (y == 0.0) return x / y;
    double mod = fmod(x, y);
    double div = (x - mod) / y;
    if (mod != 0.0 && ((y < 0.0) != (mod < 0.0))) div -= 1.0;
    if (div == 0.0) return copysign(0.0, x / y);
    double fd = floor(div);
    if (div - fd > 0.5) fd += 1.0;
    return fd;
}
__device__ __forceinline__ float _jt_floor_mod_f32(float x, float y) {
    float mod = fmodf(x, y);
    if (y == 0.0f) return mod;
    if (mod != 0.0f) {
        if ((y < 0.0f) != (mod < 0.0f)) mod += y;
    } else {
        mod = copysignf(0.0f, y);
    }
    return mod;
}
__device__ __forceinline__ double _jt_floor_mod_f64(double x, double y) {
    double mod = fmod(x, y);
    if (y == 0.0) return mod;
    if (mod != 0.0) {
        if ((y < 0.0) != (mod < 0.0)) mod += y;
    } else {
        mod = copysign(0.0, y);
    }
    return mod;
}
__device__ __forceinline__ int _jt_floor_div_i32(int x, int y) {
    if (y == 0) return 0;
    if (y == -1) return (int)(0u - (unsigned int)x);
    int q = x / y, r = x % y;
    return q - (int)(r != 0 && ((r < 0) != (y < 0)));
}
__device__ __forceinline__ long long _jt_floor_div_i64(long long x, long long y) {
    if (y == 0) return 0;
    if (y == -1) return (long long)(0ull - (unsigned long long)x);
    long long q = x / y, r = x % y;
    return q - (long long)(r != 0 && ((r < 0) != (y < 0)));
}
__device__ __forceinline__ int _jt_floor_mod_i32(int x, int y) {
    if (y == 0 || y == -1) return 0;
    int r = x % y;
    return r + (int)(r != 0 && ((r < 0) != (y < 0))) * y;
}
__device__ __forceinline__ long long _jt_floor_mod_i64(long long x, long long y) {
    if (y == 0 || y == -1) return 0;
    long long r = x % y;
    return r + (long long)(r != 0 && ((r < 0) != (y < 0))) * y;
}
__device__ __forceinline__ int _jt_fmod_i32(int x, int y) {
    if (y == 0 || y == -1) return 0;
    return x % y;
}
__device__ __forceinline__ long long _jt_fmod_i64(long long x, long long y) {
    if (y == 0 || y == -1) return 0;
    return x % y;
}
)CUDA";

// Validate eligibility and optionally build the complete snapshot.
static bool analyze_and_build(FusedOp* fop, const std::string* jit_key, NvrtcCompileSnapshot* out) {
    if (!use_nvrtc || profiler_enable || jit_search_kernel) return false;
    if (!fop || fop->ops.empty() || fop->ops.size() > 128 || fop->vars.size() > 128) return false;
    if (fop->execution_backend() != BackendId::Cuda) return false;
    if (!nvrtc_api().available) return false;

    // Loop options: allow empty or only CPU-specific "FLAGS: -O3 "
    if (jit_key && !jit_key->empty()) {
        for (const auto& kv : parse_jit_keys(*jit_key)) {
            if (kv.first == "shapes") return false;
            if (kv.first == "choices" && !kv.second.empty() && kv.second != "FLAGS: -O3 :1,")
                return false;
        }
    } else if (fop->loop_options && !fop->loop_options->empty()) {
        for (const auto& kv : *fop->loop_options) {
            if (kv.second == 0) continue;
            if (kv.first == "FLAGS: -O3 ") continue;
            return false;
        }
    }

    const int nvars = (int)fop->vars.size();
    const int nops = (int)fop->ops.size();

    // Check output shape agreement and index_t == int32
    NanoVector main_shape;
    bool has_main_shape = false;
    int main_rank = 0;
    int64 main_num = 0;

    for (int vi = 0; vi < nvars; ++vi) {
        Var* v = fop->vars[vi].var;
        int vtype = fop->vars[vi].type;
        if (!v || v->dsize() <= 0 || to_nvrtc_dtype(v->dtype()) == NvrtcDType::INVALID) return false;
        if (v->shape.size() > 8) return false;
        if (v->num < 0) return false;
        for (size_t d = 0; d < v->shape.size(); ++d) {
            if (v->shape[d] < 0 || v->shape[d] >= std::numeric_limits<int32_t>::max()) return false;
            int64 st = v->storage_stride((uint)d);
            if (st < 0 || st >= std::numeric_limits<int32_t>::max()) return false;
        }
        if (vtype != 1) {
            if (v->num >= std::numeric_limits<int32_t>::max() ||
                v->storage_span_bytes() / v->dsize() >= std::numeric_limits<int32_t>::max())
                return false;
        }
        if (vtype == 2) {
            if (!v->is_contiguous()) return false;
            if (!has_main_shape) {
                main_shape = v->shape;
                main_rank = (int)v->shape.size();
                main_num = v->num;
                has_main_shape = true;
            } else if (v->shape != main_shape) {
                return false;
            }
        }
    }
    (void)main_num;
    if (!has_main_shape) return false;

    static const OpId id_array = op_ids::array();
    static const OpId id_unary = get_op_id("unary");
    static const OpId id_binary = get_op_id("binary");
    static const OpId id_ternary = get_op_id("ternary");
    static const OpId id_broadcast = get_op_id("broadcast_to");

    // Determine structural var roles purely from the DAG (invariant under jit_key!):
    //   0 = unreached, 1 = scalar broadcast source (always 1 element), 2 = main grid.
    // This guarantees that a kernel cached from a 1-element main_shape run never
    // misclassifies a scalar BroadcastToOp input as non-scalar on a later larger run.
    std::vector<uint8_t> var_role(nvars, 0);
    for (int vi = 0; vi < nvars; ++vi) {
        if (fop->vars[vi].type == 2) var_role[vi] = 2;
    }
    auto assign_role = [&](int vi, uint8_t role) -> bool {
        if (vi < 0 || vi >= nvars) return false;
        if (var_role[vi] == 0) { var_role[vi] = role; return true; }
        return var_role[vi] == role;
    };
    for (int oi = nops - 1; oi >= 0; --oi) {
        Op* op = fop->ops[oi];
        if (!op->codegen().extra_flags.empty()) return false;
        OpId oid = op->type_id();
        if (oid == id_array) {
            auto* aop = static_cast<ArrayOp*>(op);
            if (aop->output->num != 1 || !aop->host_value_known()) return false;
            int ovi = fop->var_index.get(aop->output);
            if (ovi < 0 || var_role[ovi] == 0) return false;
        } else if (oid == id_unary) {
            auto* uop = static_cast<UnaryOp*>(op);
            if (uop->layout_value) return false;
            int xvi = fop->var_index.get(uop->x);
            int yvi = fop->var_index.get(uop->y);
            if (xvi < 0 || yvi < 0 || var_role[yvi] == 0) return false;
            if (!assign_role(xvi, var_role[yvi])) return false;
        } else if (oid == id_binary) {
            auto* bop = static_cast<BinaryOp*>(op);
            if (bop->layout_value) return false;
            int xvi = fop->var_index.get(bop->x);
            int yvi = fop->var_index.get(bop->y);
            int zvi = fop->var_index.get(bop->z);
            if (xvi < 0 || yvi < 0 || zvi < 0 || var_role[zvi] == 0) return false;
            if (!assign_role(xvi, var_role[zvi]) || !assign_role(yvi, var_role[zvi])) return false;
        } else if (oid == id_ternary) {
            auto* top = static_cast<TernaryOp*>(op);
            int cvi = fop->var_index.get(top->cond);
            int xvi = fop->var_index.get(top->x);
            int yvi = fop->var_index.get(top->y);
            int zvi = fop->var_index.get(top->z);
            if (cvi < 0 || xvi < 0 || yvi < 0 || zvi < 0 || var_role[zvi] == 0) return false;
            if (!assign_role(cvi, var_role[zvi]) || !assign_role(xvi, var_role[zvi]) ||
                !assign_role(yvi, var_role[zvi])) return false;
        } else if (oid == id_broadcast) {
            auto* btop = static_cast<BroadcastToOp*>(op);
            if (btop->expand_is_view || btop->x->num != 1) return false;
            int xvi = fop->var_index.get(btop->x);
            int zvi = fop->var_index.get(btop->z);
            if (xvi < 0 || zvi < 0 || var_role[zvi] != 2) return false;
            if (!assign_role(xvi, 1)) return false;
        } else {
            return false;
        }
    }

    // Verify every var's runtime shape matches its structural role.
    for (int vi = 0; vi < nvars; ++vi) {
        Var* v = fop->vars[vi].var;
        if (var_role[vi] == 0) return false;
        if (var_role[vi] == 1) {
            if (v->num != 1) return false;
        } else {
            if (v->shape != main_shape) return false;
        }
    }

    std::vector<int> var_in_slot(nvars, -1);
    std::vector<int> var_out_slot(nvars, -1);
    std::vector<int> op_scalar_slot(nops, -1);

    NvrtcKernelPlan plan;
    plan.rank = main_rank;

    // Bind external inputs and outputs using structural roles.
    for (int vi = 0; vi < nvars; ++vi) {
        Var* v = fop->vars[vi].var;
        int vtype = fop->vars[vi].type;
        if (vtype == 0) {
            bool scalar_in = (var_role[vi] == 1);
            bool strided = (!scalar_in) && !v->is_contiguous();
            uint32_t smask = strided ? v->stride_pattern() : 0u;
            var_in_slot[vi] = (int)plan.inputs.size();
            plan.inputs.push_back({vi, to_nvrtc_dtype(v->dtype()), strided, scalar_in, smask});
            if (strided) {
                plan.any_strided = true;
                plan.union_stride_mask |= smask;
            }
        } else if (vtype == 2) {
            var_out_slot[vi] = (int)plan.outputs.size();
            plan.outputs.push_back({vi, to_nvrtc_dtype(v->dtype()), false});
        }
    }

    if (plan.inputs.size() > 24 || plan.outputs.size() > 16) return false;
    int strided_count = 0;
    for (const auto& ib : plan.inputs) if (ib.is_strided) ++strided_count;
    if (strided_count > 12) return false;

    // Validate every member op in topological order and collect descriptors
    struct OpDesc {
        int kind = 0; // 1=array, 2=unary, 3=binary, 4=ternary, 5=broadcast
        NanoString op_ns;
        int in_vi[3] = {-1, -1, -1};
        int out_vi = -1;
        NvrtcDType in_dt[3] = {};
        NvrtcDType out_dt = NvrtcDType::INVALID;
        int scalar_slot = -1;
    };
    std::vector<OpDesc> descs;
    if (out) descs.reserve(nops);

    for (int oi = 0; oi < nops; ++oi) {
        Op* op = fop->ops[oi];
        OpId oid = op->type_id();

        if (oid == id_array) {
            auto* aop = static_cast<ArrayOp*>(op);
            int ovi = fop->var_index.get(aop->output);
            int sslot = (int)plan.scalar_arrays.size();
            NvrtcDType dt = to_nvrtc_dtype(aop->output->dtype());
            plan.scalar_arrays.push_back({oi, dt});
            op_scalar_slot[oi] = sslot;
            if (out) {
                OpDesc d;
                d.kind = 1;
                d.out_vi = ovi;
                d.out_dt = dt;
                d.scalar_slot = sslot;
                descs.push_back(d);
            }
        } else if (oid == id_unary) {
            auto* uop = static_cast<UnaryOp*>(op);
            int xvi = fop->var_index.get(uop->x);
            int yvi = fop->var_index.get(uop->y);
            NvrtcDType tx = to_nvrtc_dtype(uop->x->dtype());
            NvrtcDType ty = to_nvrtc_dtype(uop->y->dtype());
            if (!unary_op_supported(uop->ns, tx, ty)) return false;
            if (out) {
                OpDesc d;
                d.kind = 2;
                d.op_ns = uop->ns;
                d.in_vi[0] = xvi;
                d.in_dt[0] = tx;
                d.out_vi = yvi;
                d.out_dt = ty;
                descs.push_back(d);
            }
        } else if (oid == id_binary) {
            auto* bop = static_cast<BinaryOp*>(op);
            int xvi = fop->var_index.get(bop->x);
            int yvi = fop->var_index.get(bop->y);
            int zvi = fop->var_index.get(bop->z);
            NvrtcDType tx = to_nvrtc_dtype(bop->x->dtype());
            NvrtcDType ty = to_nvrtc_dtype(bop->y->dtype());
            NvrtcDType tz = to_nvrtc_dtype(bop->z->dtype());
            if (!binary_op_supported(bop->ns, tx, ty, tz)) return false;
            if (out) {
                OpDesc d;
                d.kind = 3;
                d.op_ns = bop->ns;
                d.in_vi[0] = xvi; d.in_vi[1] = yvi;
                d.in_dt[0] = tx;  d.in_dt[1] = ty;
                d.out_vi = zvi;   d.out_dt = tz;
                descs.push_back(d);
            }
        } else if (oid == id_ternary) {
            auto* top = static_cast<TernaryOp*>(op);
            int cvi = fop->var_index.get(top->cond);
            int xvi = fop->var_index.get(top->x);
            int yvi = fop->var_index.get(top->y);
            int zvi = fop->var_index.get(top->z);
            NvrtcDType tc = to_nvrtc_dtype(top->cond->dtype());
            NvrtcDType tx = to_nvrtc_dtype(top->x->dtype());
            NvrtcDType ty = to_nvrtc_dtype(top->y->dtype());
            NvrtcDType tz = to_nvrtc_dtype(top->z->dtype());
            if (tc == NvrtcDType::INVALID || tx == NvrtcDType::INVALID ||
                ty == NvrtcDType::INVALID || tz == NvrtcDType::INVALID) return false;
            if (out) {
                OpDesc d;
                d.kind = 4;
                d.in_vi[0] = cvi; d.in_vi[1] = xvi; d.in_vi[2] = yvi;
                d.in_dt[0] = tc;  d.in_dt[1] = tx;  d.in_dt[2] = ty;
                d.out_vi = zvi;   d.out_dt = tz;
                descs.push_back(d);
            }
        } else if (oid == id_broadcast) {
            auto* btop = static_cast<BroadcastToOp*>(op);
            int xvi = fop->var_index.get(btop->x);
            int zvi = fop->var_index.get(btop->z);
            if (out) {
                OpDesc d;
                d.kind = 5;
                d.in_vi[0] = xvi;
                d.in_dt[0] = to_nvrtc_dtype(btop->x->dtype());
                d.out_vi = zvi;
                d.out_dt = to_nvrtc_dtype(btop->z->dtype());
                descs.push_back(d);
            }
        } else {
            return false;
        }
    }

    if (plan.scalar_arrays.size() > 32) return false;

    // Can we emit a vectorized path?
    // - vec8 (uint4 = 16 bytes) when all non-scalar inputs/outputs are 2-byte
    //   (float16 / bfloat16), achieving full 128-bit coalesced bandwidth.
    // - vec4 when all non-scalar inputs/outputs are <= 4 bytes
    //   (float32/int32/float16/bfloat16/int8/uint8/bool -> float4/int4/ushort4/uchar4/char4).
    int vec_width = 1;
    if (!plan.any_strided) {
        bool all_vec4 = true;
        bool all_2byte = true;
        for (const auto& ib : plan.inputs) {
            if (ib.is_scalar) continue;
            if (vec4_ctype(ib.dtype) == nullptr) { all_vec4 = false; all_2byte = false; break; }
            if (nvrtc_dtype_size(ib.dtype) != 2) all_2byte = false;
        }
        for (const auto& ob : plan.outputs) {
            if (vec4_ctype(ob.dtype) == nullptr) { all_vec4 = false; all_2byte = false; break; }
            if (nvrtc_dtype_size(ob.dtype) != 2) all_2byte = false;
        }
        if (all_2byte) vec_width = 8;
        else if (all_vec4) vec_width = 4;
    }
    plan.vec_width = vec_width;

    if (!out) return true;

    // Generate pure CUDA C kernel source (ZERO #includes!)
    std::ostringstream ss;
    ss << kNvrtcPrelude << "\n";

    // Helper device function computing one element given loaded inputs and scalar args.
    // Using an inline helper avoids duplicating DAG code between vec4 and scalar tail!
    ss << "__device__ __forceinline__ void _jt_eval_elem(\n";
    bool first_arg = true;
    auto sep = [&]() { if (!first_arg) ss << ",\n"; first_arg = false; };
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        sep();
        ss << "    " << storage_ctype(plan.inputs[k].dtype) << " in_v" << k;
    }
    for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) {
        sep();
        ss << "    " << storage_ctype(plan.scalar_arrays[k].dtype) << " sc" << k;
    }
    for (size_t k = 0; k < plan.outputs.size(); ++k) {
        sep();
        ss << "    " << storage_ctype(plan.outputs[k].dtype) << "& out_v" << k;
    }
    ss << "\n) {\n";

    // Bind external inputs into v{vi}
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        int vi = plan.inputs[k].fop_var_idx;
        ss << "    " << storage_ctype(plan.inputs[k].dtype) << " v" << vi << " = in_v" << k << ";\n";
    }
    // Emit ops in topological order
    for (const auto& d : descs) {
        ss << "    " << storage_ctype(d.out_dt) << " v" << d.out_vi << " = ";
        if (d.kind == 1) {
            ss << "sc" << d.scalar_slot;
        } else if (d.kind == 2) {
            ss << emit_unary_expr(d.op_ns, "v" + std::to_string(d.in_vi[0]), d.in_dt[0], d.out_dt);
        } else if (d.kind == 3) {
            ss << emit_binary_expr(d.op_ns,
                "v" + std::to_string(d.in_vi[0]), d.in_dt[0],
                "v" + std::to_string(d.in_vi[1]), d.in_dt[1], d.out_dt);
        } else if (d.kind == 4) {
            std::string c = emit_as_bool("v" + std::to_string(d.in_vi[0]), d.in_dt[0]);
            std::string a = emit_cast("v" + std::to_string(d.in_vi[1]), d.in_dt[1], d.out_dt);
            std::string b = emit_cast("v" + std::to_string(d.in_vi[2]), d.in_dt[2], d.out_dt);
            ss << "(" << c << " ? (" << a << ") : (" << b << "))";
        } else if (d.kind == 5) {
            ss << emit_cast("v" + std::to_string(d.in_vi[0]), d.in_dt[0], d.out_dt);
        }
        ss << ";\n";
    }
    for (size_t k = 0; k < plan.outputs.size(); ++k) {
        int vi = plan.outputs[k].fop_var_idx;
        ss << "    out_v" << k << " = v" << vi << ";\n";
    }
    ss << "}\n\n";

    // Kernel signature: flat typed parameters matching launch trampoline exact order
    ss << "extern \"C\" __global__ void jittor_nvrtc_kernel(\n";
    ss << "    int num";
    if (plan.vec_width > 1) ss << ",\n    int vec_num";
    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        ss << ",\n    const " << storage_ctype(plan.inputs[k].dtype) << "* in" << k;
    }
    for (size_t k = 0; k < plan.outputs.size(); ++k) {
        ss << ",\n    " << storage_ctype(plan.outputs[k].dtype) << "* out" << k;
    }
    for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) {
        ss << ",\n    " << storage_ctype(plan.scalar_arrays[k].dtype) << " sc" << k;
    }
    if (plan.any_strided) {
        for (int d = 0; d < plan.rank; ++d) {
            if (!(plan.union_stride_mask & (1u << d))) continue;
            if (d > 0) ss << ",\n    int shape" << d;
            if (d < plan.rank - 1) ss << ",\n    int above" << d;
        }
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (!plan.inputs[k].is_strided) continue;
            for (int d = 0; d < plan.rank; ++d) {
                if (plan.inputs[k].stride_mask & (1u << d)) {
                    ss << ",\n    int stride_" << k << "_" << d;
                }
            }
        }
    }
    ss << "\n) {\n";
    ss << "    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n";
    ss << "    int gstride = (int)(gridDim.x * blockDim.x);\n";

    if (plan.vec_width == 8) {
        // 16-byte uint4 (8 x 2-byte) vectorized loop for pure FP16/BF16 kernels.
        ss << "    for (int vi = tid; vi < vec_num; vi += gstride) {\n";
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (plan.inputs[k].is_scalar) {
                ss << "        " << storage_ctype(plan.inputs[k].dtype) << " sv" << k << " = in" << k << "[0];\n";
            } else {
                ss << "        uint4 iv" << k << "_raw = reinterpret_cast<const uint4*>(in" << k << ")[vi];\n";
                ss << "        const " << storage_ctype(plan.inputs[k].dtype) << "* iv" << k
                   << " = reinterpret_cast<const " << storage_ctype(plan.inputs[k].dtype) << "*>(&iv" << k << "_raw);\n";
            }
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        uint4 ov" << k << "_raw;\n";
            ss << "        " << storage_ctype(plan.outputs[k].dtype) << "* ov" << k
               << " = reinterpret_cast<" << storage_ctype(plan.outputs[k].dtype) << "*>(&ov" << k << "_raw);\n";
        }
        for (int lane = 0; lane < 8; ++lane) {
            ss << "        _jt_eval_elem(";
            bool f = true;
            auto s = [&]() { if (!f) ss << ", "; f = false; };
            for (size_t k = 0; k < plan.inputs.size(); ++k) {
                s();
                if (plan.inputs[k].is_scalar) ss << "sv" << k;
                else ss << "iv" << k << "[" << lane << "]";
            }
            for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) {
                s();
                ss << "sc" << k;
            }
            for (size_t k = 0; k < plan.outputs.size(); ++k) {
                s();
                ss << "ov" << k << "[" << lane << "]";
            }
            ss << ");\n";
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        reinterpret_cast<uint4*>(out" << k << ")[vi] = ov" << k << "_raw;\n";
        }
        ss << "    }\n";
        ss << "    for (int i = vec_num * 8 + tid; i < num; i += gstride) {\n";
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            ss << "        " << storage_ctype(plan.inputs[k].dtype) << " iv" << k
               << " = in" << k << (plan.inputs[k].is_scalar ? "[0];\n" : "[i];\n");
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        " << storage_ctype(plan.outputs[k].dtype) << " ov" << k << ";\n";
        }
        ss << "        _jt_eval_elem(";
        {
            bool f = true;
            auto s = [&]() { if (!f) ss << ", "; f = false; };
            for (size_t k = 0; k < plan.inputs.size(); ++k) { s(); ss << "iv" << k; }
            for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) { s(); ss << "sc" << k; }
            for (size_t k = 0; k < plan.outputs.size(); ++k) { s(); ss << "ov" << k; }
        }
        ss << ");\n";
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        out" << k << "[i] = ov" << k << ";\n";
        }
        ss << "    }\n";
    } else if (plan.vec_width == 4) {
        // Vectorized 4-element-per-thread path (128-bit for f32/i32, 64-bit for f16/bf16, 32-bit for i8/u8/bool).
        // Loads ALL inputs into registers BEFORE writing ANY output -> 100% safe with reuse_dying_inputs!
        ss << "    for (int vi = tid; vi < vec_num; vi += gstride) {\n";
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (plan.inputs[k].is_scalar) {
                ss << "        " << storage_ctype(plan.inputs[k].dtype) << " sv" << k << " = in" << k << "[0];\n";
            } else {
                const char* vt = vec4_ctype(plan.inputs[k].dtype);
                ss << "        " << vt << " iv" << k << " = reinterpret_cast<const " << vt << "*>(in" << k << ")[vi];\n";
            }
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        " << vec4_ctype(plan.outputs[k].dtype) << " ov" << k << ";\n";
        }
        const char* lanes[4] = {"x", "y", "z", "w"};
        for (int lane = 0; lane < 4; ++lane) {
            ss << "        _jt_eval_elem(";
            bool f = true;
            auto s = [&]() { if (!f) ss << ", "; f = false; };
            for (size_t k = 0; k < plan.inputs.size(); ++k) {
                s();
                if (plan.inputs[k].is_scalar) ss << "sv" << k;
                else ss << "iv" << k << "." << lanes[lane];
            }
            for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) {
                s();
                ss << "sc" << k;
            }
            for (size_t k = 0; k < plan.outputs.size(); ++k) {
                s();
                ss << "ov" << k << "." << lanes[lane];
            }
            ss << ");\n";
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            const char* vt = vec4_ctype(plan.outputs[k].dtype);
            ss << "        reinterpret_cast<" << vt << "*>(out" << k << ")[vi] = ov" << k << ";\n";
        }
        ss << "    }\n";
        // Scalar tail (also serves full array when vec_num == 0 due to unaligned ptr or num < 4)
        ss << "    for (int i = vec_num * 4 + tid; i < num; i += gstride) {\n";
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            ss << "        " << storage_ctype(plan.inputs[k].dtype) << " iv" << k
               << " = in" << k << (plan.inputs[k].is_scalar ? "[0];\n" : "[i];\n");
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        " << storage_ctype(plan.outputs[k].dtype) << " ov" << k << ";\n";
        }
        ss << "        _jt_eval_elem(";
        {
            bool f = true;
            auto s = [&]() { if (!f) ss << ", "; f = false; };
            for (size_t k = 0; k < plan.inputs.size(); ++k) { s(); ss << "iv" << k; }
            for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) { s(); ss << "sc" << k; }
            for (size_t k = 0; k < plan.outputs.size(); ++k) { s(); ss << "ov" << k; }
        }
        ss << ");\n";
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        out" << k << "[i] = ov" << k << ";\n";
        }
        ss << "    }\n";
    } else {
        // General scalar grid-stride path (handles strided/broadcast inputs and f64/i64 cleanly)
        ss << "    for (int i = tid; i < num; i += gstride) {\n";
        if (plan.any_strided) {
            for (int d = 0; d < plan.rank; ++d) {
                if (!(plan.union_stride_mask & (1u << d))) continue;
                ss << "        int c" << d << " = ";
                if (plan.rank == 1) ss << "i";
                else if (d == 0) ss << "(i / above0)";
                else if (d == plan.rank - 1) ss << "(i % shape" << d << ")";
                else ss << "((i / above" << d << ") % shape" << d << ")";
                ss << ";\n";
            }
        }
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (plan.inputs[k].is_scalar) {
                ss << "        " << storage_ctype(plan.inputs[k].dtype) << " iv" << k << " = in" << k << "[0];\n";
            } else if (!plan.inputs[k].is_strided) {
                ss << "        " << storage_ctype(plan.inputs[k].dtype) << " iv" << k << " = in" << k << "[i];\n";
            } else {
                ss << "        int off" << k << " = 0";
                for (int d = 0; d < plan.rank; ++d) {
                    if (plan.inputs[k].stride_mask & (1u << d)) {
                        ss << " + c" << d << " * stride_" << k << "_" << d;
                    }
                }
                ss << ";\n";
                ss << "        " << storage_ctype(plan.inputs[k].dtype) << " iv" << k << " = in" << k << "[off" << k << "];\n";
            }
        }
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        " << storage_ctype(plan.outputs[k].dtype) << " ov" << k << ";\n";
        }
        ss << "        _jt_eval_elem(";
        {
            bool f = true;
            auto s = [&]() { if (!f) ss << ", "; f = false; };
            for (size_t k = 0; k < plan.inputs.size(); ++k) { s(); ss << "iv" << k; }
            for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) { s(); ss << "sc" << k; }
            for (size_t k = 0; k < plan.outputs.size(); ++k) { s(); ss << "ov" << k; }
        }
        ss << ");\n";
        for (size_t k = 0; k < plan.outputs.size(); ++k) {
            ss << "        out" << k << "[i] = ov" << k << ";\n";
        }
        ss << "    }\n";
    }
    ss << "}\n";

    out->jit_key = *jit_key;
    out->cuda_source = ss.str();
    out->math_policy = extract_math_policy(*jit_key);
    out->sm_arch = detect_sm_arch();
    out->plan = std::move(plan);
    return out->sm_arch > 0;
}

static uint64_t fnv1a_64(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct NvrtcDiskHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t sm_arch;
    uint32_t is_cubin;
    uint64_t source_hash;
    uint64_t image_size;
};

static bool load_cached_image(const std::string& path, int sm_arch, uint64_t src_hash,
                              std::vector<char>& image, bool& is_cubin) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    NvrtcDiskHeader hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!ifs || hdr.magic != kNvrtcCacheMagic || hdr.version != kNvrtcCacheVersion ||
        hdr.sm_arch != (uint32_t)sm_arch || hdr.source_hash != src_hash ||
        hdr.image_size == 0 || hdr.image_size > (64u << 20)) {
        return false;
    }
    image.resize((size_t)hdr.image_size);
    ifs.read(image.data(), (std::streamsize)hdr.image_size);
    if (!ifs || (size_t)ifs.gcount() != (size_t)hdr.image_size) {
        image.clear();
        return false;
    }
    is_cubin = hdr.is_cubin != 0;
    return true;
}

static void save_cached_image(const std::string& path, int sm_arch, uint64_t src_hash,
                              const std::vector<char>& image, bool is_cubin) {
    if (path.empty() || image.empty()) return;
    static std::atomic<uint64_t> tmp_seq{0};
    std::string tmp_path = path + ".tmp." + std::to_string(getpid()) + "."
        + std::to_string(tmp_seq.fetch_add(1, std::memory_order_relaxed));
    NvrtcDiskHeader hdr{};
    hdr.magic = kNvrtcCacheMagic;
    hdr.version = kNvrtcCacheVersion;
    hdr.sm_arch = (uint32_t)sm_arch;
    hdr.is_cubin = is_cubin ? 1u : 0u;
    hdr.source_hash = src_hash;
    hdr.image_size = (uint64_t)image.size();
    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs) return;
        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        ofs.write(image.data(), (std::streamsize)image.size());
        ofs.flush();
        if (!ofs) {
            std::remove(tmp_path.c_str());
            return;
        }
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
    }
}

#endif // IS_CUDA && !IS_COREX && !_WIN32

} // namespace

struct NvrtcFusedKernel {
    NvrtcKernelPlan plan;
    std::vector<char> image;
    bool is_cubin = true;
    int sm_arch = 0;

#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    struct DeviceFunc {
        CUmodule mod = nullptr;
        CUfunction func = nullptr;
    };
    std::mutex mu;
    // Keyed by CUcontext so multi-GPU and context switches stay 100% isolated.
    // Modules live for the process lifetime (matching dlopen'd JIT .so handles)
    // so unloading can never race with in-flight stream launches or exit teardown.
    std::unordered_map<CUcontext, DeviceFunc> loaded;

    CUfunction get_or_load_for_current_context(bool allow_load = true) {
        auto& api = nvrtc_api();
        CUcontext ctx = nullptr;
        if (api.cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) {
            if (!allow_load) return nullptr;
            LOGf << "NVRTC launch failed: no active CUDA context on calling thread";
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = loaded.find(ctx);
            if (it != loaded.end()) return it->second.func;
        }
        if (!allow_load) return nullptr;
        if (PREDICT_BRANCH_NOT_TAKEN(graph_capture_recording)) {
            // cuModuleLoadDataEx is disallowed during CUDA stream capture; mark
            // the recording retryable so the caller runs once outside capture
            // (loading the module) and captures cleanly on the next step.
            graph_capture_refuse("NVRTC module load during graph capture", true);
            return nullptr;
        }
        CUmodule mod = nullptr;
        CUfunction fn = nullptr;
        CUresult rc = CUDA_SUCCESS;
        CUresult rc_fn = CUDA_SUCCESS;
        {
            // Release the Python GIL across driver module loading: CUPTI's
            // background callback thread holds CUPTI's internal lock while
            // acquiring the GIL (PyGILState_Ensure), and cuModuleLoadDataEx
            // acquires CUPTI's lock inside libcuda.so.1.
            GILReleaseScope gil_release;
            rc = api.cuModuleLoadDataEx(&mod, image.data(), 0, nullptr, nullptr);
            if (rc == CUDA_SUCCESS && mod) {
                rc_fn = api.cuModuleGetFunction(&fn, mod, "jittor_nvrtc_kernel");
            }
        }
        if (rc != CUDA_SUCCESS || !mod) {
            const char* err = "unknown";
            if (api.cuGetErrorString) api.cuGetErrorString(rc, &err);
            LOGf << "cuModuleLoadDataEx failed for NVRTC kernel (code=" >> rc << "):" << err;
        }
        if (rc_fn != CUDA_SUCCESS || !fn) {
            const char* err = "unknown";
            if (api.cuGetErrorString) api.cuGetErrorString(rc_fn, &err);
            LOGf << "cuModuleGetFunction failed for NVRTC kernel (code=" >> rc_fn << "):" << err;
        }
        std::lock_guard<std::mutex> lock(mu);
        auto& slot = loaded[ctx];
        if (!slot.func) {
            slot.mod = mod;
            slot.func = fn;
        }
        return slot.func;
    }

    void preload_if_context_active() {
        if (graph_capture_recording) return;
        auto& api = nvrtc_api();
        if (!api.available) return;
        CUcontext ctx = nullptr;
        if (api.cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) return;
        get_or_load_for_current_context(true);
    }
#endif
};

bool nvrtc_is_available() {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    return nvrtc_api().available;
#else
    return false;
#endif
}

bool nvrtc_can_compile(FusedOp* fop) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    return analyze_and_build(fop, nullptr, nullptr);
#else
    (void)fop;
    return false;
#endif
}

bool nvrtc_build_snapshot(FusedOp* fop, const std::string& jit_key, NvrtcCompileSnapshot& out) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    return analyze_and_build(fop, &jit_key, &out);
#else
    (void)fop; (void)jit_key; (void)out;
    return false;
#endif
}

std::shared_ptr<NvrtcFusedKernel> nvrtc_compile_snapshot(const NvrtcCompileSnapshot& snap) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    auto& api = nvrtc_api();
    if (!api.available || snap.sm_arch <= 0 || snap.cuda_source.empty()) return nullptr;

    std::string cache_tag = snap.math_policy + "|arch=" + std::to_string(snap.sm_arch) + "\n" + snap.cuda_source;
    uint64_t src_hash = fnv1a_64(cache_tag.data(), cache_tag.size());
    std::string disk_path;
    if (!cache_path.empty() && !snap.jit_key.empty()) {
        disk_path = Op::get_filename_from_jit_key(snap.jit_key, ".nvrtc.cubin");
    }

    auto kernel = std::make_shared<NvrtcFusedKernel>();
    kernel->plan = snap.plan;
    kernel->sm_arch = snap.sm_arch;

    if (!disk_path.empty() &&
        load_cached_image(disk_path, snap.sm_arch, src_hash, kernel->image, kernel->is_cubin)) {
        g_nvrtc_disk_hit_count.fetch_add(1, std::memory_order_relaxed);
        LOGvv << "NVRTC disk cache hit:" << disk_path;
        return kernel;
    }

    nvrtcProgram prog = nullptr;
    nvrtcResult rc = api.nvrtcCreateProgram(
        &prog, snap.cuda_source.c_str(), "jittor_nvrtc.cu", 0, nullptr, nullptr);
    if (rc != NVRTC_SUCCESS || !prog) {
        g_nvrtc_fallback_count.fetch_add(1, std::memory_order_relaxed);
        LOGv << "nvrtcCreateProgram failed with code" << rc << ", falling back to nvcc";
        return nullptr;
    }

    const bool emit_cubin = (api.nvrtcGetCUBINSize != nullptr && api.nvrtcGetCUBIN != nullptr);
    std::string arch_opt = std::string("--gpu-architecture=") +
        (emit_cubin ? "sm_" : "compute_") + std::to_string(snap.sm_arch);

    std::vector<const char*> opts;
    opts.push_back(arch_opt.c_str());
    if (snap.math_policy == "strict") {
        opts.push_back("--fmad=false");
        opts.push_back("--prec-div=true");
        opts.push_back("--prec-sqrt=true");
    } else if (snap.math_policy == "fast" || snap.math_policy == "default") {
        opts.push_back("--use_fast_math");
    }

    {
        GILReleaseScope gil_release;
        rc = api.nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    }
    if (rc != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        std::string log_msg;
        if (api.nvrtcGetProgramLogSize(prog, &log_sz) == NVRTC_SUCCESS && log_sz > 1) {
            log_msg.resize(log_sz);
            api.nvrtcGetProgramLog(prog, &log_msg[0]);
        }
        api.nvrtcDestroyProgram(&prog);
        g_nvrtc_fallback_count.fetch_add(1, std::memory_order_relaxed);
        LOGv << "nvrtcCompileProgram failed (code=" >> rc << "), falling back to nvcc. Log:\n" << log_msg;
        return nullptr;
    }

    size_t img_sz = 0;
    if (emit_cubin) {
        rc = api.nvrtcGetCUBINSize(prog, &img_sz);
        if (rc == NVRTC_SUCCESS && img_sz > 0) {
            kernel->image.resize(img_sz);
            rc = api.nvrtcGetCUBIN(prog, kernel->image.data());
            kernel->is_cubin = true;
        }
    } else {
        rc = api.nvrtcGetPTXSize(prog, &img_sz);
        if (rc == NVRTC_SUCCESS && img_sz > 0) {
            kernel->image.resize(img_sz);
            rc = api.nvrtcGetPTX(prog, kernel->image.data());
            kernel->is_cubin = false;
        }
    }
    api.nvrtcDestroyProgram(&prog);

    if (rc != NVRTC_SUCCESS || kernel->image.empty()) {
        g_nvrtc_fallback_count.fetch_add(1, std::memory_order_relaxed);
        LOGv << "nvrtcGetCUBIN/PTX failed (code=" >> rc << "), falling back to nvcc";
        return nullptr;
    }

    g_nvrtc_compile_count.fetch_add(1, std::memory_order_relaxed);
    if (!disk_path.empty()) {
        save_cached_image(disk_path, snap.sm_arch, src_hash, kernel->image, kernel->is_cubin);
    }
    LOGvv << "NVRTC compiled CUBIN (" >> kernel->image.size() << "bytes) for" << snap.jit_key;
    return kernel;
#else
    (void)snap;
    return nullptr;
#endif
}

void nvrtc_fused_entry(Op* op) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    auto* fop = static_cast<FusedOp*>(op);
    ASSERT(fop->context && fop->context->nvrtc_kernel);
    auto& kernel = *fop->context->nvrtc_kernel;
    const auto& plan = kernel.plan;

    // Output 0 defines the element count and shape (all outputs share shape)
    Var* ref_out = fop->vars[plan.outputs[0].fop_var_idx].var;
    int num = (int)ref_out->num;
    if (num == 0) return;

    // Collect pointers and verify alignment for vectorized path
    void* in_ptrs[24];
    void* out_ptrs[16];
    bool aligned_vec = (plan.vec_width > 1) && (num >= plan.vec_width);

    for (size_t k = 0; k < plan.inputs.size(); ++k) {
        Var* v = fop->vars[plan.inputs[k].fop_var_idx].var;
        void* p = v->ptr<void>();
        in_ptrs[k] = p;
        if (aligned_vec && !plan.inputs[k].is_scalar) {
            size_t align_req = (size_t)nvrtc_dtype_size(plan.inputs[k].dtype) * (size_t)plan.vec_width;
            if ((reinterpret_cast<uintptr_t>(p) & (align_req - 1u)) != 0) aligned_vec = false;
        }
    }
    for (size_t k = 0; k < plan.outputs.size(); ++k) {
        Var* v = fop->vars[plan.outputs[k].fop_var_idx].var;
        void* p = v->ptr<void>();
        out_ptrs[k] = p;
        if (aligned_vec) {
            size_t align_req = (size_t)nvrtc_dtype_size(plan.outputs[k].dtype) * (size_t)plan.vec_width;
            if ((reinterpret_cast<uintptr_t>(p) & (align_req - 1u)) != 0) aligned_vec = false;
        }
    }

    int vec_num = aligned_vec ? (num / plan.vec_width) : 0;

    // Read scalar ArrayOp payloads (up to 8 bytes each)
    alignas(8) char scalar_buf[32][8];
    for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) {
        auto* aop = static_cast<ArrayOp*>(fop->ops[plan.scalar_arrays[k].fop_op_idx]);
        int ds = nvrtc_dtype_size(plan.scalar_arrays[k].dtype);
        std::memcpy(scalar_buf[k], aop->ptr<char>(), (size_t)ds);
    }

    // Stride metadata if any input is strided
    int shapes[8] = {};
    int aboves[8] = {};
    int stride_vals[12 * 8];
    int stride_val_count = 0;

    if (plan.any_strided) {
        for (int d = 0; d < plan.rank; ++d) shapes[d] = (int)ref_out->shape[d];
        int acc = 1;
        for (int d = plan.rank - 1; d >= 0; --d) {
            aboves[d] = acc;
            acc *= shapes[d];
        }
        for (size_t k = 0; k < plan.inputs.size(); ++k) {
            if (!plan.inputs[k].is_strided) continue;
            Var* v = fop->vars[plan.inputs[k].fop_var_idx].var;
            for (int d = 0; d < plan.rank; ++d) {
                if (plan.inputs[k].stride_mask & (1u << d)) {
                    stride_vals[stride_val_count++] = (int)v->storage_stride((uint)d);
                }
            }
        }
    }

    // Assemble kernel parameter pointer array matching exact kernel parameter order
    void* params[256];
    int pi = 0;
    params[pi++] = &num;
    if (plan.vec_width > 1) params[pi++] = &vec_num;
    for (size_t k = 0; k < plan.inputs.size(); ++k) params[pi++] = &in_ptrs[k];
    for (size_t k = 0; k < plan.outputs.size(); ++k) params[pi++] = &out_ptrs[k];
    for (size_t k = 0; k < plan.scalar_arrays.size(); ++k) params[pi++] = scalar_buf[k];
    if (plan.any_strided) {
        for (int d = 0; d < plan.rank; ++d) {
            if (!(plan.union_stride_mask & (1u << d))) continue;
            if (d > 0) params[pi++] = &shapes[d];
            if (d < plan.rank - 1) params[pi++] = &aboves[d];
        }
        for (int si = 0; si < stride_val_count; ++si) {
            params[pi++] = &stride_vals[si];
        }
    }
    ASSERT(pi <= 256);

    CUfunction func = kernel.get_or_load_for_current_context();
    if (!func) return; // Capture refused as retryable; caller will re-run outside capture.
    int dev = current_device();
    CUstream stream = reinterpret_cast<CUstream>(
        backend_stream({BackendId::Cuda, dev >= 0 ? dev : 0}, BackendStreamKind::Compute).handle);

    const unsigned int block = 256u;
    int work_items = (plan.vec_width > 1 && vec_num > 0) ? vec_num : num;
    unsigned int grid = (unsigned int)((work_items + (int)block - 1) / (int)block);
    if (grid == 0) grid = 1;
    if (grid > 65535u) grid = 65535u;

    auto& api = nvrtc_api();
    CUresult rc = api.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, stream, params, nullptr);
    if (rc != CUDA_SUCCESS) {
        const char* err = "unknown";
        if (api.cuGetErrorString) api.cuGetErrorString(rc, &err);
        LOGf << "cuLaunchKernel failed for NVRTC fused kernel (code=" >> rc << "):" << err;
    }
    g_nvrtc_launch_count.fetch_add(1, std::memory_order_relaxed);
#else
    (void)op;
    LOGf << "nvrtc_fused_entry called without CUDA NVRTC support";
#endif
}

bool nvrtc_try_compile(FusedOp* fop, FusedOpContext& ctx, const std::string& jit_key) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    NvrtcCompileSnapshot snap;
    if (!nvrtc_build_snapshot(fop, jit_key, snap)) return false;
    auto kernel = nvrtc_compile_snapshot(snap);
    if (!kernel) return false;
    ctx.nvrtc_kernel = std::move(kernel);
    ctx.entry = &nvrtc_fused_entry;
    return true;
#else
    (void)fop; (void)ctx; (void)jit_key;
    return false;
#endif
}

void nvrtc_preload_kernel(NvrtcFusedKernel& kernel) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    kernel.preload_if_context_active();
#else
    (void)kernel;
#endif
}

bool nvrtc_try_compile_and_run(FusedOp* fop, JK& jk) {
#if defined(IS_CUDA) && !defined(IS_COREX) && !defined(_WIN32)
    if (!use_nvrtc || profiler_enable || jit_search_kernel || graph_capture_recording) return false;
    if (!nvrtc_can_compile(fop)) return false;

    std::string jit_key = jk.to_string();
    auto ctx = std::make_shared<FusedOpContext>();
    {
        // Charge in-process NVRTC first-run compilation to compile_ns rather
        // than operator_launch in jt.profile / torch.profiler.
        StepTraceCompileScope trace_compile;
        if (!nvrtc_try_compile(fop, *ctx, jit_key)) return false;
        if (ctx->nvrtc_kernel) ctx->nvrtc_kernel->preload_if_context_active();
    }

    fop->context_owner = ctx;
    fop->context = ctx.get();
    jit_fused_ops.set(jit_key, ctx);
    jit_key_mapper.set(jit_key, jit_key);
    Profiler::record_and_run(ctx->entry, fop, jit_key.c_str());
    return true;
#else
    (void)fop; (void)jk;
    return false;
#endif
}

std::vector<int64> nvrtc_stats() {
    return {
        nvrtc_is_available() ? 1 : 0,
        g_nvrtc_compile_count.load(std::memory_order_relaxed),
        g_nvrtc_disk_hit_count.load(std::memory_order_relaxed),
        g_nvrtc_launch_count.load(std::memory_order_relaxed),
        g_nvrtc_fallback_count.load(std::memory_order_relaxed),
    };
}

} // namespace jittor
