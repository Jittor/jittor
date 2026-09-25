// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// cuDNN's fused scaled-dot-product attention, behind a C interface.
//
// The cuDNN frontend is header-only C++17; Jittor's JIT compiles operators as
// C++14. So this file is built on its own, once, into a shared library that
// `cudnn_attention_cuda.py` loads with RTLD_GLOBAL, and the `jt.code`
// operators that run inside the executor only call the plain functions
// below. cuDNN itself is not linked here: Jittor has already loaded it
// globally, and the undefined symbols resolve against that copy.
//
// Layout is [batch, heads, seq, head_dim], dense. Key/value may have fewer
// heads than the query (grouped-query attention). An additive bias of shape
// [bias_b, bias_h, seq_q, seq_kv] -- bias_b 1 or batch, bias_h 1 or heads, 0
// for none -- carries an attention mask; the forward takes one, the backward
// does not. Graphs are built once per shape and kept: building one costs
// milliseconds.
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

namespace fe = cudnn_frontend;

namespace {

enum { DTYPE_HALF = 0, DTYPE_BF16 = 1 };

using Key = std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int, uint32_t, int64_t, int64_t, cudnnHandle_t>;

struct Forward {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, bias, o, stats;
};

struct Backward {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dout, stats, dq, dk, dv;
};

std::mutex cache_mutex;
std::map<Key, std::shared_ptr<Forward>> forward_cache;
std::map<Key, std::shared_ptr<Backward>> backward_cache;
std::map<Key, bool> unsupported;
thread_local std::string last_error;

fe::DataType_t io_type(int dtype) {
    return dtype == DTYPE_BF16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
}

std::vector<int64_t> dims(int64_t b, int64_t h, int64_t s, int64_t d) { return {b, h, s, d}; }
std::vector<int64_t> dense(int64_t h, int64_t s, int64_t d) { return {h * s * d, s * d, d, 1}; }

uint32_t bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

std::shared_ptr<fe::graph::Tensor_attributes> input(fe::graph::Graph& graph, const char* name,
        int64_t b, int64_t h, int64_t s, int64_t d) {
    return graph.tensor(fe::graph::Tensor_attributes()
        .set_name(name).set_dim(dims(b, h, s, d)).set_stride(dense(h, s, d)));
}

std::shared_ptr<fe::graph::Tensor_attributes> stats_input(fe::graph::Graph& graph,
        int64_t b, int64_t h, int64_t s) {
    return graph.tensor(fe::graph::Tensor_attributes()
        .set_name("stats").set_dim(dims(b, h, s, 1)).set_stride(dense(h, s, 1))
        .set_data_type(fe::DataType_t::FLOAT));
}

bool build(fe::graph::Graph& graph, cudnnHandle_t handle) {
    auto status = graph.build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        last_error = status.get_message();
        return false;
    }
    return true;
}

std::shared_ptr<Forward> forward_graph(cudnnHandle_t handle, int dtype, int64_t b, int64_t h,
        int64_t hk, int64_t sq, int64_t skv, int64_t d, float scale, int causal, int training,
        int64_t bias_b, int64_t bias_h) {
    Key key{0, dtype, b, h, hk, sq, skv, d, causal * 2 + training, bits(scale),
            bias_b, bias_h, handle};
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = forward_cache.find(key);
    if (found != forward_cache.end()) return found->second;
    if (unsupported.count(key)) return nullptr;
    auto entry = std::make_shared<Forward>();
    entry->graph = std::make_shared<fe::graph::Graph>();
    auto& graph = *entry->graph;
    graph.set_io_data_type(io_type(dtype))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    entry->q = input(graph, "q", b, h, sq, d);
    entry->k = input(graph, "k", b, hk, skv, d);
    entry->v = input(graph, "v", b, hk, skv, d);
    auto options = fe::graph::SDPA_attributes().set_name("sdpa")
        .set_generate_stats(training != 0).set_attn_scale(scale);
    if (causal) options.set_causal_mask(true);
    if (bias_b) {
        entry->bias = graph.tensor(fe::graph::Tensor_attributes()
            .set_name("bias").set_dim({bias_b, bias_h, sq, skv})
            .set_stride({bias_h * sq * skv, sq * skv, skv, 1}));
        options.set_bias(entry->bias);
    }
    auto outputs = graph.sdpa(entry->q, entry->k, entry->v, options);
    entry->o = outputs[0];
    entry->o->set_output(true).set_dim(dims(b, h, sq, d)).set_stride(dense(h, sq, d));
    if (training) {
        entry->stats = outputs[1];
        entry->stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
            .set_dim(dims(b, h, sq, 1)).set_stride(dense(h, sq, 1));
    }
    if (!build(graph, handle)) {
        unsupported[key] = true;
        return nullptr;
    }
    forward_cache[key] = entry;
    return entry;
}

std::shared_ptr<Backward> backward_graph(cudnnHandle_t handle, int dtype, int64_t b, int64_t h,
        int64_t hk, int64_t sq, int64_t skv, int64_t d, float scale, int causal) {
    Key key{1, dtype, b, h, hk, sq, skv, d, causal, bits(scale), 0, 0, handle};
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = backward_cache.find(key);
    if (found != backward_cache.end()) return found->second;
    if (unsupported.count(key)) return nullptr;
    auto entry = std::make_shared<Backward>();
    entry->graph = std::make_shared<fe::graph::Graph>();
    auto& graph = *entry->graph;
    graph.set_io_data_type(io_type(dtype))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    entry->q = input(graph, "q", b, h, sq, d);
    entry->k = input(graph, "k", b, hk, skv, d);
    entry->v = input(graph, "v", b, hk, skv, d);
    entry->o = input(graph, "o", b, h, sq, d);
    entry->dout = input(graph, "dout", b, h, sq, d);
    entry->stats = stats_input(graph, b, h, sq);
    auto options = fe::graph::SDPA_backward_attributes().set_name("sdpa_backward")
        .set_attn_scale(scale);
    if (causal) options.set_causal_mask(true);
    auto grads = graph.sdpa_backward(entry->q, entry->k, entry->v, entry->o, entry->dout,
                                     entry->stats, options);
    entry->dq = grads[0];
    entry->dk = grads[1];
    entry->dv = grads[2];
    entry->dq->set_output(true).set_dim(dims(b, h, sq, d)).set_stride(dense(h, sq, d));
    entry->dk->set_output(true).set_dim(dims(b, hk, skv, d)).set_stride(dense(hk, skv, d));
    entry->dv->set_output(true).set_dim(dims(b, hk, skv, d)).set_stride(dense(hk, skv, d));
    if (!build(graph, handle)) {
        unsupported[key] = true;
        return nullptr;
    }
    backward_cache[key] = entry;
    return entry;
}

int run(fe::graph::Graph& graph, cudnnHandle_t handle,
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>& pack,
        void* workspace) {
    auto status = graph.execute(handle, pack, workspace);
    if (!status.is_good()) {
        last_error = status.get_message();
        return 1;
    }
    return 0;
}

} // namespace

extern "C" {

const char* jt_cudnn_sdpa_last_error() { return last_error.c_str(); }

// Builds (and caches) the graphs a call would use; 0 if cuDNN cannot run it.
// `workspace` receives the larger of the two workspace sizes.
int jt_cudnn_sdpa_supported(cudnnHandle_t handle, int dtype, int64_t b, int64_t h, int64_t hk,
        int64_t sq, int64_t skv, int64_t d, float scale, int causal, int training,
        int64_t bias_b, int64_t bias_h, int64_t* workspace) {
    try {
        auto forward = forward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal, training,
                                     bias_b, bias_h);
        if (!forward) return 0;
        int64_t size = forward->graph->get_workspace_size();
        if (training) {
            auto backward = backward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal);
            if (!backward) return 0;
            size = std::max(size, backward->graph->get_workspace_size());
        }
        if (workspace) *workspace = size;
        return 1;
    } catch (const std::exception& e) {
        last_error = e.what();
        return 0;
    }
}

int64_t jt_cudnn_sdpa_forward_workspace(cudnnHandle_t handle, int dtype, int64_t b, int64_t h,
        int64_t hk, int64_t sq, int64_t skv, int64_t d, float scale, int causal, int training,
        int64_t bias_b, int64_t bias_h) {
    auto forward = forward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal, training,
                                 bias_b, bias_h);
    return forward ? forward->graph->get_workspace_size() : -1;
}

int jt_cudnn_sdpa_forward(cudnnHandle_t handle, int dtype, int64_t b, int64_t h, int64_t hk,
        int64_t sq, int64_t skv, int64_t d, float scale, int causal, int training,
        int64_t bias_b, int64_t bias_h,
        void* q, void* k, void* v, void* bias, void* o, void* stats, void* workspace) {
    try {
        auto forward = forward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal, training,
                                     bias_b, bias_h);
        if (!forward) return 1;
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
            {forward->q, q}, {forward->k, k}, {forward->v, v}, {forward->o, o}};
        if (training) pack[forward->stats] = stats;
        if (bias_b) pack[forward->bias] = bias;
        return run(*forward->graph, handle, pack, workspace);
    } catch (const std::exception& e) {
        last_error = e.what();
        return 1;
    }
}

int64_t jt_cudnn_sdpa_backward_workspace(cudnnHandle_t handle, int dtype, int64_t b, int64_t h,
        int64_t hk, int64_t sq, int64_t skv, int64_t d, float scale, int causal) {
    auto backward = backward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal);
    return backward ? backward->graph->get_workspace_size() : -1;
}

int jt_cudnn_sdpa_backward(cudnnHandle_t handle, int dtype, int64_t b, int64_t h, int64_t hk,
        int64_t sq, int64_t skv, int64_t d, float scale, int causal,
        void* q, void* k, void* v, void* o, void* dout, void* stats,
        void* dq, void* dk, void* dv, void* workspace) {
    try {
        auto backward = backward_graph(handle, dtype, b, h, hk, sq, skv, d, scale, causal);
        if (!backward) return 1;
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> pack = {
            {backward->q, q}, {backward->k, k}, {backward->v, v}, {backward->o, o},
            {backward->dout, dout}, {backward->stats, stats},
            {backward->dq, dq}, {backward->dk, dk}, {backward->dv, dv}};
        return run(*backward->graph, handle, pack, workspace);
    } catch (const std::exception& e) {
        last_error = e.what();
        return 1;
    }
}

} // extern "C"
