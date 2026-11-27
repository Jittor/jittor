// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/common.h"
#include "codegen/jit_key.h"

namespace jittor {

struct Op;
struct FusedOp;
struct FusedOpContext;

DECLARE_FLAG(int, use_nvrtc);

enum class NvrtcDType : uint8_t {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    I32 = 4,
    I64 = 5,
    I8 = 6,
    U8 = 7,
    BOOL = 8,
    INVALID = 255,
};

struct NvrtcInputBinding {
    int fop_var_idx = -1;
    NvrtcDType dtype = NvrtcDType::INVALID;
    bool is_strided = false;
    bool is_scalar = false;
    uint32_t stride_mask = 0;
};

struct NvrtcOutputBinding {
    int fop_var_idx = -1;
    NvrtcDType dtype = NvrtcDType::INVALID;
    bool is_scalar = false;
};

struct NvrtcScalarArrayBinding {
    int fop_op_idx = -1;
    NvrtcDType dtype = NvrtcDType::INVALID;
};

struct NvrtcKernelPlan {
    int rank = 0;
    bool any_strided = false;
    int vec_width = 1;
    uint32_t union_stride_mask = 0;
    std::vector<NvrtcInputBinding> inputs;
    std::vector<NvrtcOutputBinding> outputs;
    std::vector<NvrtcScalarArrayBinding> scalar_arrays;
};

// Self-contained compilation snapshot with ZERO Op* or Var* pointers.
// Safe to hand off to an asynchronous background compilation thread.
struct NvrtcCompileSnapshot {
    std::string jit_key;
    std::string cuda_source;
    std::string math_policy;
    int sm_arch = 0;
    NvrtcKernelPlan plan;
};

struct NvrtcFusedKernel;

// Whether libnvrtc and libcuda driver APIs are dynamically available.
bool nvrtc_is_available();

// Fast check whether a CUDA FusedOp is eligible for in-process NVRTC compilation.
bool nvrtc_can_compile(FusedOp* fop);

// Build a self-contained snapshot (ZERO Op*/Var* pointers) synchronously on the
// calling thread while fop and its vars are live.
bool nvrtc_build_snapshot(FusedOp* fop, const std::string& jit_key, NvrtcCompileSnapshot& out);

// Compile (or load from disk cache) a snapshot into an executable NvrtcFusedKernel.
// Pure CPU work (no CUDA driver context required); safe on any background thread.
std::shared_ptr<NvrtcFusedKernel> nvrtc_compile_snapshot(const NvrtcCompileSnapshot& snap);

// Compile an eligible CUDA FusedOp directly into `ctx` (used by parallel_compiler).
bool nvrtc_try_compile(FusedOp* fop, FusedOpContext& ctx, const std::string& jit_key);

// Eagerly load an NVRTC kernel's CUmodule into the calling thread's current
// CUDA context (if one is active and not recording a CUDA Graph).
void nvrtc_preload_kernel(NvrtcFusedKernel& kernel);

// Compile (or load from disk cache), cache into jit_fused_ops/jit_key_mapper,
// and execute an eligible CUDA FusedOp immediately.
bool nvrtc_try_compile_and_run(FusedOp* fop, JK& jk);

// Generic jit_op_entry_t trampoline for launching an NVRTC-compiled FusedOp.
void nvrtc_fused_entry(Op* op);

// Internal diagnostic counters for tests/benchmarks:
// [available, compile_count, disk_cache_hit_count, launch_count, fallback_count]
// @pyjt(_nvrtc_stats)
vector<int64> nvrtc_stats();

} // namespace jittor
