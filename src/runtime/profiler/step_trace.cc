// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#include <malloc.h>
#endif
#include "runtime/profiler/step_trace.h"
#include "runtime/profiler/memory_history.h"
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

std::atomic<int> step_trace_mode{0};

namespace {

std::atomic<uint64> trace_generation{0};

constexpr const char* kLayout =
    "ops:seq,kind,batch,name,t_start,t_alloc,t_end,device,fused,shapes,site,out_bytes,stack,range,"
    "t_compile,compile_ns,stacks;"
    "batches:batch,parent,t_begin,t_planned,t_compiled,t_launched,t_end;"
    "waits:t_begin,t_end,batch;"
    "memory:t,kind,device,bytes,allocator,allocation,op,var,var_desc,producer,site";
constexpr int kOpFields = 17;
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

// ---- Pure-C++ CUPTI activity collector ------------------------------------
// Callbacks run directly in C++ without acquiring the Python GIL, eliminating
// GIL-vs-CUPTI deadlocks when an 8 MiB buffer completes mid-step on CUPTI's
// background delivery thread and speeding up record decoding by ~50x.

using CuptiGetNextRecord = int (*)(uint8_t*, size_t, void**);
using CuptiGetDropped = int (*)(void*, uint32_t, size_t*);

constexpr uint32_t kCuptiKindMemcpy = 1;
constexpr uint32_t kCuptiKindMemset = 2;
constexpr uint32_t kCuptiKindConcurrentKernel = 10;
constexpr uint32_t kCuptiKindMemcpy2 = 22;
constexpr uint32_t kCuptiKindExternalCorrelation = 39;
constexpr uint32_t kCuptiExtKindCustom0 = 3;
constexpr size_t kCuptiBufferBytes = 8u << 20;
constexpr size_t kCuptiMaxSpareBuffers = 8;

template <typename T>
inline T read_unaligned(const uint8_t* base, size_t offset) {
    T val;
    std::memcpy(&val, base + offset, sizeof(T));
    return val;
}

inline uint8_t* alloc_cupti_buffer() {
#ifdef _WIN32
    return static_cast<uint8_t*>(_aligned_malloc(kCuptiBufferBytes, 8));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 8, kCuptiBufferBytes) != 0) return nullptr;
    return static_cast<uint8_t*>(ptr);
#endif
}

inline void free_cupti_buffer(uint8_t* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

struct CuptiState {
    std::atomic<CuptiGetNextRecord> get_next{nullptr};
    std::atomic<CuptiGetDropped> get_dropped{nullptr};

    std::mutex buf_mutex;
    vector<uint8_t*> spare_buffers;

    std::mutex data_mutex;
    vector<string> names;
    unordered_map<string, int64> name_ids;
    vector<int64> kernels;       // 15 int64s per kernel
    vector<int64> copies;        // 7 int64s per copy/memset
    vector<int64> correlations;  // 2 int64s per pair: (correlation_id, external_id)
    int64 dropped = 0;

    int64 intern_name_locked(const string& s) {
        auto it = name_ids.find(s);
        if (it != name_ids.end()) return it->second;
        int64 id = (int64)names.size();
        names.push_back(s);
        name_ids.emplace(s, id);
        return id;
    }

    void reset() {
        {
            std::lock_guard<std::mutex> guard(data_mutex);
            names.clear();
            name_ids.clear();
            kernels.clear();
            copies.clear();
            correlations.clear();
            dropped = 0;
        }
        std::lock_guard<std::mutex> bguard(buf_mutex);
        while (spare_buffers.size() < 4) {
            uint8_t* buf = alloc_cupti_buffer();
            if (!buf) break;
            spare_buffers.push_back(buf);
        }
    }
};

CuptiState& cupti_state() {
    static CuptiState* s = new CuptiState();  // outlives static destructors
    return *s;
}

void cupti_buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
    auto& cs = cupti_state();
    uint8_t* buf = nullptr;
    {
        std::lock_guard<std::mutex> guard(cs.buf_mutex);
        if (!cs.spare_buffers.empty()) {
            buf = cs.spare_buffers.back();
            cs.spare_buffers.pop_back();
        }
    }
    if (!buf) buf = alloc_cupti_buffer();
    *buffer = buf;
    *size = buf ? kCuptiBufferBytes : 0;
    *max_num_records = 0;
}

void cupti_buffer_completed(void* ctx, uint32_t stream_id, uint8_t* buffer,
                            size_t size, size_t valid_size) {
    (void)size;
    auto& cs = cupti_state();
    auto get_next = cs.get_next.load(std::memory_order_acquire);
    auto get_dropped = cs.get_dropped.load(std::memory_order_acquire);
    size_t local_dropped = 0;
    if (get_dropped) {
        size_t d = 0;
        if (get_dropped(ctx, stream_id, &d) == 0) local_dropped = d;
    }

    vector<string> local_names;
    unordered_map<string, int64> local_name_ids;
    vector<int64> local_kernels;
    vector<int64> local_copies;
    vector<int64> local_corrs;

    if (buffer && valid_size > 0 && get_next) {
        void* record = nullptr;
        while (get_next(buffer, valid_size, &record) == 0 && record) {
            const uint8_t* r = static_cast<const uint8_t*>(record);
            uint32_t kind = read_unaligned<uint32_t>(r, 0);
            if (kind == kCuptiKindConcurrentKernel) {
                const char* cname = read_unaligned<const char*>(r, 104);
                string sname = cname ? string(cname) : string("?");
                int64 nid;
                auto it = local_name_ids.find(sname);
                if (it != local_name_ids.end()) {
                    nid = it->second;
                } else {
                    nid = (int64)local_names.size();
                    local_names.push_back(sname);
                    local_name_ids.emplace(std::move(sname), nid);
                }
                int64 rec[15] = {
                    (int64)read_unaligned<uint64_t>(r, 16),   // start
                    (int64)read_unaligned<uint64_t>(r, 24),   // end
                    (int64)read_unaligned<uint32_t>(r, 40),   // device
                    (int64)read_unaligned<uint32_t>(r, 48),   // stream
                    (int64)read_unaligned<uint32_t>(r, 92),   // correlation
                    nid,                                      // local name id
                    (int64)read_unaligned<uint32_t>(r, 156),  // graph
                    (int64)read_unaligned<uint32_t>(r, 52),   // gridX
                    (int64)read_unaligned<uint32_t>(r, 56),   // gridY
                    (int64)read_unaligned<uint32_t>(r, 60),   // gridZ
                    (int64)read_unaligned<uint32_t>(r, 64),   // blockX
                    (int64)read_unaligned<uint32_t>(r, 68),   // blockY
                    (int64)read_unaligned<uint32_t>(r, 72),   // blockZ
                    (int64)read_unaligned<uint16_t>(r, 6),    // registersPerThread
                    (int64)read_unaligned<uint32_t>(r, 76)
                        + (int64)read_unaligned<uint32_t>(r, 80),  // static + dynamic shmem
                };
                local_kernels.insert(local_kernels.end(), rec, rec + 15);
            } else if (kind == kCuptiKindMemcpy) {
                int64 rec[7] = {
                    (int64)read_unaligned<uint64_t>(r, 16),  // start
                    (int64)read_unaligned<uint64_t>(r, 24),  // end
                    (int64)read_unaligned<uint32_t>(r, 32),  // device
                    (int64)read_unaligned<uint32_t>(r, 40),  // stream
                    (int64)read_unaligned<uint32_t>(r, 44),  // correlation
                    (int64)read_unaligned<uint8_t>(r, 4),    // copyKind (1..10)
                    (int64)read_unaligned<uint64_t>(r, 8),   // bytes
                };
                local_copies.insert(local_copies.end(), rec, rec + 7);
            } else if (kind == kCuptiKindMemset) {
                int64 rec[7] = {
                    (int64)read_unaligned<uint64_t>(r, 16),  // start
                    (int64)read_unaligned<uint64_t>(r, 24),  // end
                    (int64)read_unaligned<uint32_t>(r, 32),  // device
                    (int64)read_unaligned<uint32_t>(r, 40),  // stream
                    (int64)read_unaligned<uint32_t>(r, 44),  // correlation
                    0,                                       // 0 encodes memset
                    (int64)read_unaligned<uint64_t>(r, 8),   // bytes
                };
                local_copies.insert(local_copies.end(), rec, rec + 7);
            } else if (kind == kCuptiKindMemcpy2) {
                // CUpti_ActivityMemcpyPtoP4: copyKind@4, bytes@8, start@16,
                // end@24, stream@40, srcDeviceId@44, correlationId@60.
                uint8_t ckind = read_unaligned<uint8_t>(r, 4);
                int64 rec[7] = {
                    (int64)read_unaligned<uint64_t>(r, 16),  // start
                    (int64)read_unaligned<uint64_t>(r, 24),  // end
                    (int64)read_unaligned<uint32_t>(r, 44),  // srcDeviceId
                    (int64)read_unaligned<uint32_t>(r, 40),  // stream
                    (int64)read_unaligned<uint32_t>(r, 60),  // correlation
                    (int64)(ckind ? ckind : 10),             // default PtoP (10)
                    (int64)read_unaligned<uint64_t>(r, 8),   // bytes
                };
                local_copies.insert(local_copies.end(), rec, rec + 7);
            } else if (kind == kCuptiKindExternalCorrelation) {
                if (read_unaligned<uint32_t>(r, 4) == kCuptiExtKindCustom0) {
                    local_corrs.push_back((int64)read_unaligned<uint32_t>(r, 16));
                    local_corrs.push_back((int64)read_unaligned<uint64_t>(r, 8));
                }
            }
        }
    }

    if (buffer) {
        std::lock_guard<std::mutex> bguard(cs.buf_mutex);
        if (cs.spare_buffers.size() < kCuptiMaxSpareBuffers) {
            cs.spare_buffers.push_back(buffer);
            buffer = nullptr;
        }
    }
    if (buffer) free_cupti_buffer(buffer);

    if (!local_kernels.empty() || !local_copies.empty() || !local_corrs.empty() || local_dropped) {
        std::lock_guard<std::mutex> guard(cs.data_mutex);
        if (!local_names.empty()) {
            vector<int64> remap(local_names.size());
            for (size_t i = 0; i < local_names.size(); ++i)
                remap[i] = cs.intern_name_locked(local_names[i]);
            for (size_t i = 5; i < local_kernels.size(); i += 15)
                local_kernels[i] = remap[(size_t)local_kernels[i]];
        }
        cs.kernels.insert(cs.kernels.end(), local_kernels.begin(), local_kernels.end());
        cs.copies.insert(cs.copies.end(), local_copies.begin(), local_copies.end());
        cs.correlations.insert(cs.correlations.end(), local_corrs.begin(), local_corrs.end());
        cs.dropped += (int64)local_dropped;
    }
}

struct TraceState {
    std::mutex mutex;
    vector<string> strings;
    unordered_map<string, int64> string_ids;
    unordered_map<uint64, int64> origin_ids;
    // Op id -> the range open when it was built (step_trace_note_op_built).
    unordered_map<int64, int64> built_in_range;
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
        strings.clear(); string_ids.clear(); origin_ids.clear(); built_in_range.clear();
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
    // The Python stack that built the op (memory_history.h's stack ids), when
    // op stacks are captured; 0 otherwise.
    int64 stack = 0;
    // The record_function range open on the building thread when the op was
    // built, -1 for none: under lazy execution it runs later, outside it.
    int64 range = -1;
    // JIT compilation (or loading a cached kernel) inside the op: when it
    // first started and how long it took in all (StepTraceCompileScope).
    int64 t_compile = 0, compile_ns = 0;
    // A fused op's members' stacks, distinct and in member order, as an
    // interned "id,id,..." string; 0 for an unfused op.
    int64 stacks = 0;
    uint64 gen = 0;
    int kind = ok_op;
};

thread_local vector<PendingOp> pending_ops;
thread_local int64 current_batch = -1;
thread_local int64 current_op = -1;
thread_local Var* current_var = nullptr;
thread_local int wait_depth = 0;
thread_local int64 wait_started = 0;
thread_local uint64 wait_gen = 0;

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
    auto push = state().push.load(std::memory_order_relaxed);
    if (push) push(kCorrelationKind, (uint64)id);
}

void pop_correlation() {
    auto pop = state().pop.load(std::memory_order_relaxed);
    uint64 last = 0;
    if (pop) pop(kCorrelationKind, &last);
}

inline void purge_stale_pending(uint64 cur_gen) {
    if (pending_ops.empty()) return;
    pending_ops.erase(
        std::remove_if(pending_ops.begin(), pending_ops.end(),
                       [cur_gen](const PendingOp& p) { return p.gen != cur_gen; }),
        pending_ops.end());
    current_op = pending_ops.empty() ? -1 : pending_ops.back().seq;
}

int64 begin_record(int kind, int64 name) {
    uint64 cur_gen = trace_generation.load(std::memory_order_acquire);
    purge_stale_pending(cur_gen);
    PendingOp p;
    p.kind = kind;
    p.batch = current_batch;
    p.name = name;
    p.gen = cur_gen;
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
    uint64 cur_gen = trace_generation.load(std::memory_order_acquire);
    PendingOp p;
    bool found = false;
    for (auto it = pending_ops.end(); it != pending_ops.begin(); ) {
        --it;
        if (it->seq == seq) {
            p = *it;
            pending_ops.erase(it);
            found = true;
            break;
        }
    }
    if (found) pop_correlation();
    purge_stale_pending(cur_gen);
    if (!found || p.seq < 0 || p.gen != cur_gen || !(step_trace_mode.load(std::memory_order_relaxed) & st_ops))
        return;
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.room(s.ops, kOpFields)) return;
    int64 rec[kOpFields] = {p.seq, p.kind, p.batch, p.name, p.t_start, p.t_alloc,
                            t_end, p.device, p.fused, p.shapes, p.site, p.out_bytes, p.stack,
                            p.range, p.t_compile, p.compile_ns, p.stacks};
    s.ops.insert(s.ops.end(), rec, rec + kOpFields);
}

void record_existing_allocations() {
    // What is already live is part of every peak the trace will see, so it is
    // reported rather than left as an unexplained floor. Gather Var metadata
    // from the holder graph, then emit every occupied SFRL pool block at its
    // exact aligned block size (so views/slices, alignment padding, and
    // graph-fenced blocks never cause untracked_at_start / unattributed_at_peak
    // drift), followed by any non-SFRL existing Var allocations.
    ExecutorEntryScope entry;
    struct ExistingVarMeta {
        int64 var_id = -1;
        int64 var_size = 0;
        int device = -1;
        string desc;
        string producer;
        uint64 origin = 0;
    };
    vector<Node*> queue;
    unordered_set<Node*> seeds;
    for (auto& vh : runtime_holder_state().holders())
        if (seeds.insert(vh->var).second) queue.push_back(vh->var);
    bfs_both(queue, [](Node*){ return true; });
    std::map<pair<const void*, int64>, ExistingVarMeta> by_block;
    for (Node* node : queue) {
        if (!node->is_var()) continue;
        Var* v = (Var*)node;
        if (!v->mem_ptr || !v->allocator || v->size <= 0) continue;
        auto key = std::make_pair((const void*)v->allocator, (int64)v->allocation);
        auto it = by_block.find(key);
        // If several views alias one block, prefer the largest Var (usually
        // the owning tensor rather than a small slice).
        if (it != by_block.end() && it->second.var_size >= v->size) continue;
        ExistingVarMeta meta;
        meta.var_id = v->id;
        meta.var_size = v->size;
        meta.device = v->allocator->is_cuda() ? v->allocator->device() : -1;
        meta.desc = var_desc(v);
        Op* producer = v->input();
        if (producer) {
            meta.producer = producer->name_ex();
            meta.origin = producer->launch_origin;
        } else if (v->name.size()) {
            meta.producer = string("param:") + v->name.c_str();
        }
        by_block[key] = std::move(meta);
    }

    struct PoolBlockEntry {
        int device;
        int64 bytes;
        const void* allocator;
        int64 allocation;
    };
    vector<PoolBlockEntry> pool_blocks;
    std::set<pair<const void*, int64>> emitted;
    for (SFRLAllocator* a : SFRLAllocator::sfrl_allocators) {
        std::unique_lock<std::recursive_mutex> lock(a->mutex);
        const int dev = a->is_cuda() ? a->device() : -1;
        for (CachingBlock* b : a->id_space.occupied_id_mapper) {
            if (!b || !b->occupied) continue;
            pool_blocks.push_back({dev, (int64)b->size, (const void*)a, (int64)b->id});
        }
    }

    int64 t = now_ns();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    for (const auto& pb : pool_blocks) {
        auto key = std::make_pair(pb.allocator, pb.allocation);
        emitted.insert(key);
        int64 var_id = -1, desc_id = 0, prod_id = 0, site_id = -1;
        auto it = by_block.find(key);
        if (it != by_block.end()) {
            var_id = it->second.var_id;
            desc_id = s.intern(it->second.desc);
            prod_id = it->second.producer.empty() ? 0 : s.intern(it->second.producer);
            site_id = s.site(it->second.origin);
        }
        int64 rec[kMemFields] = {t, stm_existing, pb.device, pb.bytes,
            (int64)(uintptr_t)pb.allocator, pb.allocation, -1, var_id,
            desc_id, prod_id, site_id};
        if (!s.room(s.memory, kMemFields)) return;
        s.memory.insert(s.memory.end(), rec, rec + kMemFields);
    }
    // Non-SFRL existing Vars (e.g., when use_sfrl_allocator=0).
    for (const auto& kv : by_block) {
        if (emitted.count(kv.first)) continue;
        const auto& meta = kv.second;
        int64 rec[kMemFields] = {t, stm_existing, meta.device, meta.var_size,
            (int64)(uintptr_t)kv.first.first, kv.first.second, -1, meta.var_id,
            s.intern(meta.desc),
            meta.producer.empty() ? 0 : s.intern(meta.producer),
            s.site(meta.origin)};
        if (!s.room(s.memory, kMemFields)) return;
        s.memory.insert(s.memory.end(), rec, rec + kMemFields);
    }
}

} // namespace

// ---- Python API ----------------------------------------------------------

void step_trace_start(int64 mode) {
    trace_generation.fetch_add(1, std::memory_order_acq_rel);
    auto& s = state();
    {
        std::lock_guard<std::mutex> guard(s.mutex);
        s.clear();
    }
    if (mode & st_memory) record_existing_allocations();
    step_trace_mode.store((int)mode, std::memory_order_release);
}

void step_trace_stop() {
    step_trace_mode.store(0, std::memory_order_release);
    trace_generation.fetch_add(1, std::memory_order_acq_rel);
}

void step_trace_clear() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    s.clear();
}

int64 step_trace_active() { return step_trace_mode.load(std::memory_order_relaxed); }
int64 step_trace_now() { return now_ns(); }

void step_trace_set_correlation(int64 push_fn, int64 pop_fn) {
    auto& s = state();
    s.push.store((CorrelationPush)(uintptr_t)push_fn, std::memory_order_release);
    s.pop.store((CorrelationPop)(uintptr_t)pop_fn, std::memory_order_release);
}

vector<int64> step_trace_cupti_init(int64 get_next_record_fn, int64 get_dropped_fn) {
    auto& cs = cupti_state();
    cs.get_next.store((CuptiGetNextRecord)(uintptr_t)get_next_record_fn, std::memory_order_release);
    cs.get_dropped.store((CuptiGetDropped)(uintptr_t)get_dropped_fn, std::memory_order_release);
    return {
        (int64)(uintptr_t)&cupti_buffer_requested,
        (int64)(uintptr_t)&cupti_buffer_completed,
    };
}

void step_trace_cupti_reset() {
    cupti_state().reset();
}

vector<string> step_trace_cupti_take_names() {
    auto& cs = cupti_state();
    std::lock_guard<std::mutex> guard(cs.data_mutex);
    vector<string> out;
    out.swap(cs.names);
    cs.name_ids.clear();
    return out;
}

vector<int64> step_trace_cupti_take_kernels() {
    auto& cs = cupti_state();
    std::lock_guard<std::mutex> guard(cs.data_mutex);
    vector<int64> out;
    out.swap(cs.kernels);
    return out;
}

vector<int64> step_trace_cupti_take_copies() {
    auto& cs = cupti_state();
    std::lock_guard<std::mutex> guard(cs.data_mutex);
    vector<int64> out;
    out.swap(cs.copies);
    return out;
}

vector<int64> step_trace_cupti_take_correlations() {
    auto& cs = cupti_state();
    std::lock_guard<std::mutex> guard(cs.data_mutex);
    vector<int64> out;
    out.swap(cs.correlations);
    return out;
}

int64 step_trace_cupti_dropped() {
    auto& cs = cupti_state();
    std::lock_guard<std::mutex> guard(cs.data_mutex);
    int64 d = cs.dropped;
    cs.dropped = 0;
    return d;
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
    if (!(step_trace_mode.load(std::memory_order_relaxed) & st_ops)) return -1;
    int64 sid;
    {
        auto& s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        sid = s.intern(name);
    }
    int64 seq = begin_record(ok_range, sid);
    if (PREDICT_BRANCH_NOT_TAKEN(capture_op_stacks.load(std::memory_order_relaxed)))
        if (PendingOp* p = find_pending(seq)) p->stack = capture_memory_stack();
    return seq;
}

void step_trace_range_end(int64 id) {
    if (id < 0) return;
    end_record(id);
}

int64 graph_launch_count() { return graph_launches.load(std::memory_order_relaxed); }

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
        std::unique_lock<std::recursive_mutex> lock(a->mutex);
        temp_used += a->used_memory;
        temp_reserved += a->used_memory + a->unused_memory;
    }
    return {used, reserved, sfrl_device_peak_bytes(device),
            sfrl_device_reserved_peak_bytes(device), free_blocks, largest_free,
            temp_used, temp_reserved};
}

// ---- hooks ---------------------------------------------------------------

StepTraceOpScope::StepTraceOpScope() {
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops))
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
    const bool with_shapes = step_trace_mode.load(std::memory_order_relaxed) & st_shapes;
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
    auto built = s.built_in_range.find(root->id);
    if (built != s.built_in_range.end()) {
        p->range = built->second;
    } else if (is_fused) {
        auto* fop = (FusedOp*)op;
        for (auto it = fop->ops.rbegin(); it != fop->ops.rend(); ++it) {
            auto mb = s.built_in_range.find((*it)->id);
            if (mb != s.built_in_range.end()) { p->range = mb->second; break; }
        }
    }
    p->name = s.intern(name);
    p->shapes = with_shapes ? s.intern(shapes) : 0;
    p->site = s.site(root->launch_origin);
    p->stack = root->memory_stack;
    if (is_fused) {
        // Each member was built on its own line; the root's stack names one.
        string ids;
        vector<uint32> seen;
        for (Op* member : ((FusedOp*)op)->ops) {
            uint32 id = member->memory_stack;
            if (!id || std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
            seen.push_back(id);
            if (ids.size()) ids += ',';
            ids += std::to_string(id);
        }
        p->stacks = s.intern(ids);
    }
    p->device = device;
    p->out_bytes = out_bytes;
}

StepTraceCompileScope::StepTraceCompileScope() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops)
            || current_op < 0)
        return;
    // Only the op's own compile: one a batch nested inside it does while
    // planning is that batch's compile phase.
    PendingOp* p = find_pending(current_op);
    if (!p || p->kind != ok_op || p->batch != current_batch) return;
    seq = current_op;
    t_begin = now_ns();
}

StepTraceCompileScope::~StepTraceCompileScope() {
    if (seq < 0) return;
    PendingOp* p = find_pending(seq);
    if (!p) return;
    if (!p->t_compile) p->t_compile = t_begin;
    p->compile_ns += now_ns() - t_begin;
}

void StepTraceOpScope::allocated() {
    if (seq < 0) return;
    if (PendingOp* p = find_pending(seq)) p->t_alloc = now_ns();
}

StepTraceBatchScope::StepTraceBatchScope() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops)) return;
    gen = trace_generation.load(std::memory_order_acquire);
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.room(s.batches, kBatchFields)) return;
    prev_batch = current_batch;
    batch = s.next_batch++;
    int64 t = now_ns();
    int64 rec[kBatchFields] = {batch, prev_batch, t, 0, 0, 0, 0};
    s.batches.insert(s.batches.end(), rec, rec + kBatchFields);
    current_batch = batch;
}

void StepTraceBatchScope::mark(int phase) {
    if (batch < 0 || gen != trace_generation.load(std::memory_order_acquire)) return;
    auto& s = state();
    int64 t = now_ns();
    std::lock_guard<std::mutex> guard(s.mutex);
    size_t at = (size_t)batch * kBatchFields;
    if (at + kBatchFields > s.batches.size() || s.batches[at] != batch) return;
    s.batches[at + 2 + phase] = t;
}

StepTraceBatchScope::~StepTraceBatchScope() {
    if (batch < 0) return;
    current_batch = prev_batch;
    if (gen != trace_generation.load(std::memory_order_acquire)) return;
    auto& s = state();
    int64 t = now_ns();
    std::lock_guard<std::mutex> guard(s.mutex);
    size_t at = (size_t)batch * kBatchFields;
    if (at + kBatchFields <= s.batches.size() && s.batches[at] == batch) {
        s.batches[at + 6] = t;
    }
}

void step_trace_wait_begin() {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops)) return;
    uint64 cur_gen = trace_generation.load(std::memory_order_acquire);
    if (wait_depth == 0 || wait_gen != cur_gen) {
        wait_depth = 1;
        wait_started = now_ns();
        wait_gen = cur_gen;
    } else {
        wait_depth++;
    }
}

void step_trace_wait_end() {
    if (wait_depth <= 0) return;
    wait_depth--;
    if (wait_depth > 0) return;
    uint64 cur_gen = trace_generation.load(std::memory_order_acquire);
    int64 started = wait_started;
    wait_started = 0;
    if (!started || wait_gen != cur_gen
            || !PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops))
        return;
    int64 t = now_ns();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (s.room(s.waits, kWaitFields)) {
        int64 rec[kWaitFields] = {started, t, current_batch};
        s.waits.insert(s.waits.end(), rec, rec + kWaitFields);
    }
}

void step_trace_mem(int kind, int device, int64 bytes, const void* allocator, int64 allocation) {
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_memory)) return;
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
    graph_launches.fetch_add(1, std::memory_order_relaxed);
    if (PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops)) {
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
    if (!PREDICT_BRANCH_NOT_TAKEN(step_trace_mode.load(std::memory_order_relaxed) & st_ops)) return;
    for (auto it = pending_ops.rbegin(); it != pending_ops.rend(); ++it)
        if (it->kind == ok_graph_launch) { end_record(it->seq); return; }
}

Var* step_trace_current_var() { return current_var; }

void step_trace_note_op_built(Op* op) {
    uint64 cur_gen = trace_generation.load(std::memory_order_acquire);
    for (auto it = pending_ops.rbegin(); it != pending_ops.rend(); ++it) {
        if (it->kind != ok_range || it->gen != cur_gen) continue;
        auto& s = state();
        std::lock_guard<std::mutex> guard(s.mutex);
        if (s.built_in_range.size() < kMaxRecords) s.built_in_range[op->id] = it->seq;
        return;
    }
}

StepTraceVarScope::StepTraceVarScope(Var* v) {
    if (PREDICT_BRANCH_NOT_TAKEN((step_trace_mode.load(std::memory_order_relaxed) & st_memory)
                                 | memory_history_mode.load(std::memory_order_relaxed))) {
        on = true;
        saved = current_var;
        current_var = v;
    }
}

StepTraceVarScope::~StepTraceVarScope() {
    if (on) current_var = saved;
}

} // jittor
