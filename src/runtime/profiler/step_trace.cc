// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <atomic>
#include <chrono>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "runtime/profiler/step_trace.h"
#include "runtime/launch_diagnostics.h"
#include "runtime/executor_entry.h"
#include "runtime/holder_state.h"
#include "core/op.h"
#include "core/var.h"
#include "core/fused_op.h"
#include "core/graph.h"
#include "core/var_holder.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/temp_allocator.h"

namespace jittor {

int step_trace_mode = 0;

namespace {

constexpr const char* kLayout =
    "ops:seq,kind,batch,name,t_start,t_alloc,t_end,device,fused,shapes,site,out_bytes;"
    "batches:batch,parent,t_begin,t_planned,t_compiled,t_launched,t_end;"
    "waits:t_begin,t_end,batch;"
    "memory:t,kind,device,bytes,allocator,allocation,op,var,var_desc,producer,site";
constexpr int kOpFields = 12;
constexpr int kBatchFields = 7;
constexpr int kWaitFields = 3;
constexpr int kMemFields = 11;
// A trace that outgrows this many records of one kind is truncated rather
// than allowed to eat the host; `step_trace_layout` says so, and so does the
// Python report.
constexpr size_t kMaxRecords = 1u << 20;

enum OpKind { ok_op = 0, ok_graph_launch = 1, ok_range = 2 };

using CorrelationPush = int (*)(int, uint64);
using CorrelationPop = int (*)(int, uint64*);
constexpr int kCorrelationKind = 3;   // CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0

struct TraceState {
    std::mutex mutex;
    vector<string> strings;
    unordered_map<string, int64> string_ids;
    unordered_map<uint64, int64> origin_ids;
    vector<int64> ops, batches, waits, memory;
    int64 next_seq = 0;
    int64 next_batch = 0;
    bool truncated = false;
    std::atomic<CorrelationPush> push{nullptr};
    std::atomic<CorrelationPop> pop{nullptr};

    int64 intern(const string& s) {
        auto it = string_ids.find(s);
        if (it != string_ids.end()) return it->second;
        int64 id = strings.size();
        strings.push_back(s);
        string_ids.emplace(s, id);
        return id;
    }

    int64 site(uint64 origin) {
        if (!origin) return -1;
        auto it = origin_ids.find(origin);
        if (it != origin_ids.end()) return it->second;
        LaunchOrigin where;
        int64 id = -1;
        if (runtime_launch_history().origin(origin, where)) {
            std::stringstream ss;
            ss << where.file << ':' << where.line;
            id = intern(ss.str());
        }
        origin_ids.emplace(origin, id);
        return id;
    }

    bool room(const vector<int64>& table, int fields) {
        if (table.size() / fields < kMaxRecords) return true;
        truncated = true;
        return false;
    }

    void clear() {
        strings.clear(); string_ids.clear(); origin_ids.clear();
        ops.clear(); batches.clear(); waits.clear(); memory.clear();
        next_seq = next_batch = 0;
        truncated = false;
        intern("");   // id 0 is the empty string
    }
};

TraceState& state() {
    static TraceState* s = new TraceState();   // outlives static destructors
    return *s;
}

std::atomic<int64> graph_launches{0};

struct PendingOp {
    int64 seq = -1, batch = -1, name = 0, t_start = 0, t_alloc = 0;
    int64 device = -1, fused = 0, shapes = 0, site = -1, out_bytes = 0;
    int kind = ok_op;
};

thread_local vector<PendingOp> pending_ops;
thread_local int64 current_batch = -1;
thread_local int64 current_op = -1;
thread_local Var* current_var = nullptr;
thread_local int64 wait_started = 0;

inline int64 now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

string var_desc(Var* v) {
    std::stringstream ss;
    ss << v->dtype() << '[';
    for (int i = 0; i < (int)v->shape.size(); i++) {
        if (i) ss << ',';
        ss << v->shape[i];
    }
    ss << ']';
    return ss.str();
}

// "fused(binary.multiply x3, unary.erf, reduce.add)": the members that do
// the work, in order, with repeats folded. Inputs materialized into the
// kernel (array) and broadcasts are left out unless nothing else is there;
// they are how an expression is spelled, not what it computes.
string fused_name(const vector<Op*>& ops) {
    vector<pair<string, int>> parts;
    auto add = [&](const string& n) {
        if (parts.size() && parts.back().first == n) parts.back().second++;
        else parts.emplace_back(n, 1);
    };
    for (Op* member : ops) {
        string n = member->name();
        if (n == "array" || n == "broadcast_to") continue;
        add(member->name_ex());
    }
    if (parts.empty())
        for (Op* member : ops) add(member->name_ex());
    string name = "fused(";
    for (size_t i = 0; i < parts.size(); i++) {
        if (i == 6) { name += ", +" + std::to_string(parts.size() - i) + " more"; break; }
        if (i) name += ", ";
        name += parts[i].first;
        if (parts[i].second > 1) name += " x" + std::to_string(parts[i].second);
    }
    return name + ")";
}

void push_correlation(int64 id) {
    auto push = state().push.load();
    if (push) push(kCorrelationKind, (uint64)id);
}

void pop_correlation() {
    auto pop = state().pop.load();
    uint64 last = 0;
    if (pop) pop(kCorrelationKind, &last);
}

int64 begin_record(int kind, int64 name) {
    PendingOp p;
    p.kind = kind;
    p.batch = current_batch;
    p.name = name;
    {
        auto& s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        p.seq = s.next_seq++;
    }
    p.t_start = now_ns();
    p.t_alloc = p.t_start;
    pending_ops.push_back(p);
    push_correlation(p.seq);
    current_op = p.seq;
    return p.seq;
}

PendingOp* find_pending(int64 seq) {
    for (auto it = pending_ops.rbegin(); it != pending_ops.rend(); ++it)
        if (it->seq == seq) return &*it;
    return nullptr;
}

void end_record(int64 seq) {
    int64 t_end = now_ns();
    pop_correlation();
    PendingOp p;
    for (auto it = pending_ops.end(); it != pending_ops.begin(); ) {
        --it;
        if (it->seq == seq) { p = *it; pending_ops.erase(it); break; }
    }
    current_op = pending_ops.empty() ? -1 : pending_ops.back().seq;
    if (p.seq < 0) return;
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.room(s.ops, kOpFields)) return;
    int64 rec[kOpFields] = {p.seq, p.kind, p.batch, p.name, p.t_start, p.t_alloc,
                            t_end, p.device, p.fused, p.shapes, p.site, p.out_bytes};
    s.ops.insert(s.ops.end(), rec, rec + kOpFields);
}

void record_existing_allocations() {
    // What is already live is part of every peak the trace will see, so it is
    // reported rather than left as an unexplained floor. Reachable from the
    // holders, as the legacy memory profiler walks it.
    ExecutorEntryScope entry;
    vector<Node*> queue;
    unordered_set<Node*> seeds;
    for (auto& vh : runtime_holder_state().holders())
        if (seeds.insert(vh->var).second) queue.push_back(vh->var);
    bfs_both(queue, [](Node*){ return true; });
    std::set<pair<const void*, int64>> seen;
    int64 t = now_ns();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    for (Node* node : queue) {
        if (!node->is_var()) continue;
        Var* v = (Var*)node;
        if (!v->mem_ptr || !v->allocator || v->size <= 0) continue;
        auto key = std::make_pair((const void*)v->allocator, (int64)v->allocation);
        if (!seen.insert(key).second) continue;
        Op* producer = v->input();
        int64 rec[kMemFields] = {t, stm_existing, v->allocator->device(), v->size,
            (int64)(uintptr_t)v->allocator, (int64)v->allocation, -1, v->id,
            s.intern(var_desc(v)),
            producer ? s.intern(producer->name_ex()) : (v->name.size() ? s.intern(string("param:") + v->name.c_str()) : 0),
            producer ? s.site(producer->launch_origin) : -1};
        if (!s.room(s.memory, kMemFields)) return;
        s.memory.insert(s.memory.end(), rec, rec + kMemFields);
    }
}

} // namespace

// ---- Python API ----------------------------------------------------------

void step_trace_start(int64 mode) {
    auto& s = state();
    {
        std::lock_guard<std::mutex> guard(s.mutex);
        s.clear();
    }
    if (mode & st_memory) record_existing_allocations();
    step_trace_mode = (int)mode;
}

void step_trace_stop() { step_trace_mode = 0; }

void step_trace_clear() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    s.clear();
}

int64 step_trace_active() { return step_trace_mode; }
int64 step_trace_now() { return now_ns(); }

void step_trace_set_correlation(int64 push_fn, int64 pop_fn) {
    auto& s = state();
    s.push.store((CorrelationPush)(uintptr_t)push_fn);
    s.pop.store((CorrelationPop)(uintptr_t)pop_fn);
}

vector<string> step_trace_strings() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    return s.strings;
}

vector<int64> step_trace_ops() {
    auto& s = state(); std::lock_guard<std::mutex> guard(s.mutex); return s.ops;
}
vector<int64> step_trace_batches() {
    auto& s = state(); std::lock_guard<std::mutex> guard(s.mutex); return s.batches;
}
vector<int64> step_trace_waits() {
    auto& s = state(); std::lock_guard<std::mutex> guard(s.mutex); return s.waits;
}
vector<int64> step_trace_memory() {
    auto& s = state(); std::lock_guard<std::mutex> guard(s.mutex); return s.memory;
}

string step_trace_layout() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    return string(kLayout) + (s.truncated ? ";truncated:1" : ";truncated:0");
}

int64 step_trace_range_begin(const string& name) {
    if (!(step_trace_mode & st_ops)) return -1;
    int64 sid;
    {
        auto& s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        sid = s.intern(name);
    }
    return begin_record(ok_range, sid);
}

void step_trace_range_end(int64 id) {
    if (id < 0) return;
    end_record(id);
}

int64 graph_launch_count() { return graph_launches.load(); }

vector<int64> device_pool_stats(int device) {
    int64 used = 0, reserved = 0, free_blocks = 0, largest_free = 0;
    const bool host = device < 0;
    for (auto* a : SFRLAllocator::sfrl_allocators) {
        if (a->is_cuda() == host) continue;
        if (!host && a->device() != device) continue;
        std::unique_lock<std::recursive_mutex> lock(a->mutex);
        used += a->used_memory;
        reserved += a->used_memory + a->unused_memory;
        for (auto* pool : {&a->small_blocks, &a->large_blocks})
            for (auto& kv : pool->blocks) {
                free_blocks++;
                largest_free = std::max<int64>(largest_free, kv.second->size);
            }
    }
    int64 temp_used = 0, temp_reserved = 0;
    for (auto* a : TempAllocator::temp_allocators) {
        if (a->is_cuda() == host) continue;
        if (!host && a->device() != device) continue;
        temp_used += a->used_memory;
        temp_reserved += a->used_memory + a->unused_memory;
    }
    return {used, reserved, sfrl_device_peak_bytes(device),
            sfrl_device_reserved_peak_bytes(device), free_blocks, largest_free,
            temp_used, temp_reserved};
}

// ---- hooks ---------------------------------------------------------------

StepTraceOpScope::StepTraceOpScope() {
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops))
        seq = begin_record(ok_op, 0);
}

StepTraceOpScope::~StepTraceOpScope() {
    if (seq >= 0) end_record(seq);
}

void StepTraceOpScope::named(Op* op, bool is_fused, int device) {
    if (seq < 0) return;
    PendingOp* p = find_pending(seq);
    if (!p) return;
    string name;
    string shapes;
    int64 out_bytes = 0;
    Op* root = op;
    const bool with_shapes = step_trace_mode & st_shapes;
    if (is_fused) {
        auto* fop = (FusedOp*)op;
        p->fused = fop->ops.size();
        name = fused_name(fop->ops);
        if (fop->ops.size()) root = fop->ops.back();
        for (auto& vi : fop->vars) {
            if (vi.type == 0 && with_shapes) {
                if (shapes.size()) shapes += ", ";
                shapes += var_desc(vi.var);
            }
            if (vi.type == 2) out_bytes += vi.var->size;
        }
    } else {
        name = op->name_ex();
        if (with_shapes)
            for (Var* v : op->inputs()) {
                if (shapes.size()) shapes += ", ";
                shapes += var_desc(v);
            }
        for (Var* v : op->outputs()) out_bytes += std::max<int64>(v->size, 0);
    }
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    p->name = s.intern(name);
    p->shapes = with_shapes ? s.intern(shapes) : 0;
    p->site = s.site(root->launch_origin);
    p->device = device;
    p->out_bytes = out_bytes;
}

void StepTraceOpScope::allocated() {
    if (seq < 0) return;
    if (PendingOp* p = find_pending(seq)) p->t_alloc = now_ns();
}

StepTraceBatchScope::StepTraceBatchScope() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops)) return;
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.room(s.batches, kBatchFields)) return;
    batch = s.next_batch++;
    int64 t = now_ns();
    int64 rec[kBatchFields] = {batch, current_batch, t, 0, 0, 0, 0};
    s.batches.insert(s.batches.end(), rec, rec + kBatchFields);
    current_batch = batch;
}

void StepTraceBatchScope::mark(int phase) {
    if (batch < 0) return;
    auto& s = state();
    int64 t = now_ns();
    std::lock_guard<std::mutex> guard(s.mutex);
    size_t at = (size_t)batch * kBatchFields;
    if (at + kBatchFields > s.batches.size() || s.batches[at] != batch) return;
    s.batches[at + 2 + phase] = t;
}

StepTraceBatchScope::~StepTraceBatchScope() {
    if (batch < 0) return;
    auto& s = state();
    int64 t = now_ns();
    std::lock_guard<std::mutex> guard(s.mutex);
    size_t at = (size_t)batch * kBatchFields;
    if (at + kBatchFields <= s.batches.size() && s.batches[at] == batch) {
        s.batches[at + 6] = t;
        current_batch = s.batches[at + 1];
    } else {
        current_batch = -1;
    }
}

void step_trace_wait_begin() {
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops)) wait_started = now_ns();
}

void step_trace_wait_end() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops) || !wait_started) return;
    int64 t = now_ns();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (s.room(s.waits, kWaitFields)) {
        int64 rec[kWaitFields] = {wait_started, t, current_batch};
        s.waits.insert(s.waits.end(), rec, rec + kWaitFields);
    }
    wait_started = 0;
}

void step_trace_mem(int kind, int device, int64 bytes, const void* allocator, int64 allocation) {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_memory)) return;
    int64 t = now_ns();
    auto& s = state();
    Var* v = (kind == stm_pool && bytes > 0) ? current_var : nullptr;
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.room(s.memory, kMemFields)) return;
    int64 var_id = -1, desc = 0, producer = 0, site = -1;
    if (v) {
        var_id = v->id;
        desc = s.intern(var_desc(v));
        if (Op* op = v->input()) {
            producer = s.intern(op->name_ex());
            site = s.site(op->launch_origin);
        }
    }
    int64 rec[kMemFields] = {t, kind, device, bytes, (int64)(uintptr_t)allocator,
                             allocation, current_op, var_id, desc, producer, site};
    s.memory.insert(s.memory.end(), rec, rec + kMemFields);
}

void note_graph_launch_begin() {
    graph_launches++;
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops)) {
        int64 sid, site;
        // The Python line that asked for the launch, by the same rule as an
        // operator's `launch_origin`.
        uint64 origin = capture_launch_origin();
        {
            auto& s = state();
            std::lock_guard<std::mutex> guard(s.mutex);
            sid = s.intern("cuda_graph_launch");
            site = s.site(origin);
        }
        int64 seq = begin_record(ok_graph_launch, sid);
        if (PendingOp* p = find_pending(seq)) p->site = site;
    }
}

void note_graph_launch_end() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_ops)) return;
    for (auto it = pending_ops.rbegin(); it != pending_ops.rend(); ++it)
        if (it->kind == ok_graph_launch) { end_record(it->seq); return; }
}

StepTraceVarScope::StepTraceVarScope(Var* v) {
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode & st_memory)) {
        on = true;
        saved = current_var;
        current_var = v;
    }
}

StepTraceVarScope::~StepTraceVarScope() {
    if (on) current_var = saved;
}

} // jittor
