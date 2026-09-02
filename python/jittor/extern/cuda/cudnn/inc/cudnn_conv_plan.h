// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// Backend (graph API) execution of 2-D convolutions with a plan cache.
//
// The legacy cuDNN API redoes its heuristics and builds an execution plan
// inside every call: on cuDNN 8.9 cudnnGetConvolution*WorkspaceSize plus
// cudnnConvolution* cost about 110us of CPU per call, whether or not the
// descriptors are reused. The backend API builds the plan once per
// (kind, shapes, strides, dtypes, parameters, numerics) and executes it for
// about 12us. Each convolution op tries this path first and keeps its legacy
// code as the fallback for anything the backend declines.
//
// Header-only on purpose: only ops/*op.cc are compiled into the cuDNN op
// library, and a static inside an inline function is a single instance
// across those translation units.
#include <cudnn.h>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "cudnn_wrapper.h"
#include "executor.h"
#include "mem/mem_info.h"

namespace jittor {

enum ConvPlanKind { CONV_PLAN_FWD = 0, CONV_PLAN_BWD_DATA = 1, CONV_PLAN_BWD_FILTER = 2 };

// All int64 so the struct has no padding: its bytes are the cache key.
struct ConvPlanRequest {
    int64 kind;
    int64 dtype_x, dtype_w, dtype_y;
    int64 xdim[4], xstride[4], wdim[4], wstride[4], ydim[4], ystride[4];
    int64 pad[2], stride[2], dilation[2];
    int64 allow_tf32;   // tensor-op numerics for fp32 operands
    int64 benchmark;    // time the candidate plans instead of trusting heuristic order
};

struct ConvPlanRequestHash {
    size_t operator()(const ConvPlanRequest& r) const {
        const unsigned char* p = (const unsigned char*)&r;
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(r); i++) { h ^= p[i]; h *= 1099511628211ull; }
        return (size_t)h;
    }
};
struct ConvPlanRequestEq {
    bool operator()(const ConvPlanRequest& a, const ConvPlanRequest& b) const {
        return memcmp(&a, &b, sizeof(a)) == 0;
    }
};

struct ConvPlanEntry {
    cudnnBackendDescriptor_t plan = nullptr;
    int64 workspace = 0;
    // Descriptors the plan was built from, kept alive with it.
    std::vector<cudnnBackendDescriptor_t> owned;
    bool valid = false;
};

inline std::unordered_map<ConvPlanRequest, ConvPlanEntry, ConvPlanRequestHash, ConvPlanRequestEq>&
conv_plan_cache() {
    static std::unordered_map<ConvPlanRequest, ConvPlanEntry, ConvPlanRequestHash, ConvPlanRequestEq> cache;
    return cache;
}

// Strides of a KCRS filter stored as oihw (NCHW) or ohwi (NHWC).
inline void conv_plan_filter_strides(int64* s, const int* d, bool nhwc) {
    if (nhwc) { s[0] = (int64)d[1]*d[2]*d[3]; s[1] = 1; s[2] = (int64)d[3]*d[1]; s[3] = d[1]; }
    else      { s[0] = (int64)d[1]*d[2]*d[3]; s[1] = (int64)d[2]*d[3]; s[2] = d[3]; s[3] = 1; }
}

namespace conv_plan_detail {

struct Desc {
    cudnnBackendDescriptor_t d = nullptr;
    bool ok = true;
    explicit Desc(cudnnBackendDescriptorType_t t) {
        ok = cudnnBackendCreateDescriptor(t, &d) == CUDNN_STATUS_SUCCESS;
    }
    ~Desc() { if (d) cudnnBackendDestroyDescriptor(d); }
    cudnnBackendDescriptor_t release() { auto r = d; d = nullptr; return r; }
    void set(cudnnBackendAttributeName_t n, cudnnBackendAttributeType_t t, int64 count, const void* v) {
        if (ok) ok = cudnnBackendSetAttribute(d, n, t, count, (void*)v) == CUDNN_STATUS_SUCCESS;
    }
    void finalize() { if (ok) ok = cudnnBackendFinalize(d) == CUDNN_STATUS_SUCCESS; }
};

inline bool tensor(Desc& t, int64 uid, int64 dtype, const int64* dims, const int64* strides) {
    cudnnDataType_t dt = (cudnnDataType_t)dtype; int64 align = 16;
    t.set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
    t.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
    t.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
    t.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    t.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
    t.finalize();
    return t.ok;
}

// Engines that need runtime compilation are never used; for fp32 operands
// without tensor-op permission, engines that use tensor cores or down-convert
// inputs are skipped so the numerics match the legacy FMA path.
inline bool engine_allowed(cudnnBackendDescriptor_t cfg, bool fp32, bool allow_tf32) {
    Desc eng(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    int64_t n = 0;
    if (!eng.ok || cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE,
            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n, &eng.d) != CUDNN_STATUS_SUCCESS)
        return false;
    cudnnBackendBehaviorNote_t behavior[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT];
    if (cudnnBackendGetAttribute(eng.d, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE,
            CUDNN_BEHAVIOR_NOTE_TYPE_COUNT, &n, behavior) == CUDNN_STATUS_SUCCESS)
        for (int64_t i = 0; i < n; i++)
            if (behavior[i] == CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION) return false;
    if (fp32 && !allow_tf32) {
        cudnnBackendNumericalNote_t numerical[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
        if (cudnnBackendGetAttribute(eng.d, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE,
                CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &n, numerical) == CUDNN_STATUS_SUCCESS)
            for (int64_t i = 0; i < n; i++)
                if (numerical[i] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE
                    || numerical[i] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS) return false;
    }
    return true;
}

inline bool execute(cudnnBackendDescriptor_t plan, void* x, void* w, void* y, void* ws) {
    Desc vp(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    void* ptrs[] = {x, w, y}; int64 uids[] = {1, 2, 3};
    vp.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
    vp.set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
    vp.set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
    vp.finalize();
    if (!vp.ok) return false;
    return cudnnBackendExecute(cudnn_handle, plan, vp.d) == CUDNN_STATUS_SUCCESS;
}

// Builds the plan for a request. Candidate engine configurations come from
// the heuristics (mode B, then A, then the fallback list); with benchmark
// set, up to six candidates are timed on the real buffers, as the legacy
// cudnnFind path does.
inline void build(const ConvPlanRequest& r, ConvPlanEntry& e, void* x, void* w, void* y) {
    e.valid = false;
    bool fp32 = r.dtype_x == CUDNN_DATA_FLOAT;
    Desc X(CUDNN_BACKEND_TENSOR_DESCRIPTOR), W(CUDNN_BACKEND_TENSOR_DESCRIPTOR), Y(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    if (!tensor(X, 1, r.dtype_x, r.xdim, r.xstride)) return;
    if (!tensor(W, 2, r.dtype_w, r.wdim, r.wstride)) return;
    if (!tensor(Y, 3, r.dtype_y, r.ydim, r.ystride)) return;
    Desc conv(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    cudnnDataType_t comp = CUDNN_DATA_FLOAT; cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION; int64 nd = 2;
    conv.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
    conv.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
    conv.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nd);
    conv.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, r.pad);
    conv.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, r.pad);
    conv.set(CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, r.dilation);
    conv.set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, r.stride);
    conv.finalize();
    if (!conv.ok) return;
    cudnnBackendDescriptorType_t optype; cudnnBackendAttributeName_t ax, aw, ay, aconv, aalpha, abeta;
    if (r.kind == CONV_PLAN_FWD) {
        optype = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
        ax = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X; aw = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W;
        ay = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y; aconv = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC;
        aalpha = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA; abeta = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA;
    } else if (r.kind == CONV_PLAN_BWD_DATA) {
        optype = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
        ax = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX; aw = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W;
        ay = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY; aconv = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC;
        aalpha = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA; abeta = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA;
    } else {
        optype = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
        ax = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X; aw = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW;
        ay = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY; aconv = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC;
        aalpha = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA; abeta = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA;
    }
    Desc op(optype);
    float alpha = 1, beta = 0;
    op.set(ax, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &X.d);
    op.set(aw, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &W.d);
    op.set(ay, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &Y.d);
    op.set(aconv, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv.d);
    op.set(aalpha, CUDNN_TYPE_FLOAT, 1, &alpha);
    op.set(abeta, CUDNN_TYPE_FLOAT, 1, &beta);
    op.finalize();
    if (!op.ok) return;
    Desc graph(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    graph.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn_handle);
    graph.set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op.d);
    graph.finalize();
    if (!graph.ok) return;

    int64 limit = (int64)(mem_info.total_cuda_ram * max_workspace_ratio);
    cudnnBackendHeurMode_t modes[] = {CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK};
    struct Candidate { cudnnBackendDescriptor_t plan; int64 ws; };
    std::vector<Candidate> candidates;
    int max_candidates = r.benchmark ? 6 : 1;
    for (auto mode : modes) {
        Desc heur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        heur.set(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph.d);
        heur.set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
        heur.finalize();
        if (!heur.ok) continue;
        int64_t count = 0;
        if (cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                0, &count, nullptr) != CUDNN_STATUS_SUCCESS || count <= 0) continue;
        std::vector<cudnnBackendDescriptor_t> cfgs(count, nullptr);
        for (auto& c : cfgs) cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &c);
        int64_t got = 0;
        cudnnBackendGetAttribute(heur.d, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
            count, &got, cfgs.data());
        for (int64_t i = 0; i < got && (int)candidates.size() < max_candidates; i++) {
            if (!engine_allowed(cfgs[i], fp32, r.allow_tf32)) continue;
            Desc plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn_handle);
            plan.set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfgs[i]);
            plan.finalize();
            if (!plan.ok) continue;
            int64_t n = 0, ws = 0;
            if (cudnnBackendGetAttribute(plan.d, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64,
                    1, &n, &ws) != CUDNN_STATUS_SUCCESS || ws > limit) continue;
            candidates.push_back({plan.release(), ws});
        }
        for (auto c : cfgs) if (c) cudnnBackendDestroyDescriptor(c);
        if (candidates.size()) break;
    }
    if (candidates.empty()) return;
    int best = 0;
    if (candidates.size() > 1) {
        // Measure like cudnnFind: warm once, time twice, keep the fastest.
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        float best_ms = 1e30f;
        for (size_t i = 0; i < candidates.size(); i++) {
            size_t allocation; void* ws = nullptr;
            if (candidates[i].ws) ws = exe.temp_allocator->alloc(candidates[i].ws, allocation);
            float ms = 1e30f;
            if (execute(candidates[i].plan, x, w, y, ws)) {
                cudaEventRecord(start);
                bool ok = execute(candidates[i].plan, x, w, y, ws) && execute(candidates[i].plan, x, w, y, ws);
                cudaEventRecord(stop); cudaEventSynchronize(stop);
                if (ok) cudaEventElapsedTime(&ms, start, stop);
            }
            if (ws) exe.temp_allocator->free(ws, candidates[i].ws, allocation);
            if (ms < best_ms) { best_ms = ms; best = (int)i; }
        }
        cudaEventDestroy(start); cudaEventDestroy(stop);
        if (best_ms >= 1e30f) {
            for (auto& c : candidates) cudnnBackendDestroyDescriptor(c.plan);
            return;
        }
    }
    for (size_t i = 0; i < candidates.size(); i++)
        if ((int)i != best) cudnnBackendDestroyDescriptor(candidates[i].plan);
    e.plan = candidates[best].plan;
    e.workspace = candidates[best].ws;
    e.owned = {X.release(), W.release(), Y.release(), conv.release(), op.release(), graph.release()};
    e.valid = true;
}

} // conv_plan_detail

// Runs the convolution through a cached backend plan. Returns false when the
// backend declined the request (also cached), and the caller must then use
// its legacy path.
inline bool cudnn_conv_backend_run(const ConvPlanRequest& r, void* x, void* w, void* y) {
    if (r.dtype_x == CUDNN_DATA_DOUBLE || (((uintptr_t)x | (uintptr_t)w | (uintptr_t)y) & 15))
        return false;
    auto& cache = conv_plan_cache();
    auto it = cache.find(r);
    if (it == cache.end()) {
        if (cache.size() >= 4096) return false;
        it = cache.emplace(r, ConvPlanEntry()).first;
        conv_plan_detail::build(r, it->second, x, w, y);
    }
    auto& e = it->second;
    if (!e.valid) return false;
    size_t allocation; void* ws = nullptr;
    if (e.workspace) ws = exe.temp_allocator->alloc(e.workspace, allocation);
    bool ok = conv_plan_detail::execute(e.plan, x, w, y, ws);
    if (ws) exe.temp_allocator->free(ws, e.workspace, allocation);
    ASSERT(ok) << "cuDNN backend convolution execution failed for kind" << r.kind;
    return true;
}

} // jittor
