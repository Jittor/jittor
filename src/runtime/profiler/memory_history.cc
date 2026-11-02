// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include "runtime/profiler/memory_history.h"
#include "runtime/profiler/step_trace.h"
#include "core/op.h"
#include "core/var.h"
#include "mem/allocator/sfrl_allocator.h"

namespace jittor {

int memory_history_mode = 0;

namespace {

constexpr const char* kLayout =
    "trace:action,addr,size,stream,time_us,stack,device_free;"
    "segment:address,total_size,allocated_size,requested_size,stream,is_small,stack,blocks;"
    "block:address,size,requested_size,state,stack";
constexpr int kTraceFields = 7;
constexpr int kSegmentFields = 8;
constexpr int kBlockFields = 5;

struct Entry {
    int64 action, addr, size, stream, time_us, stack, device_free;
};

struct Live {
    int64 requested;
    uint32 stack;
};

struct HistoryState {
    std::mutex mutex;
    int64 max_entries = 0;
    // frame id -> (file, line, name), flattened; id 0 is "no frame". Frames
    // and stacks are never dropped: an Op built under one recording keeps its
    // stack id into the next.
    vector<string> frames{"", "0", ""};
    unordered_map<string, uint32> frame_ids;
    // stack id -> frame ids, innermost first; id 0 is "no stack".
    vector<vector<uint32>> stacks{{}};
    unordered_map<string, uint32> stack_ids;
    std::map<int, std::deque<Entry>> traces;
    // What each live block and segment was allocated with, by address. Device
    // addresses are unique across devices (unified addressing).
    unordered_map<uint64, Live> blocks, segments;

    void clear() {
        traces.clear();
        blocks.clear();
        segments.clear();
    }

    void push(int device, const Entry& e) {
        if (!(memory_history_mode & mh_trace)) return;
        auto& trace = traces[device];
        trace.push_back(e);
        while (max_entries > 0 && (int64)trace.size() > max_entries) trace.pop_front();
    }
};

HistoryState& state() {
    static HistoryState* s = new HistoryState();   // outlives static destructors
    return *s;
}

std::atomic<MemoryStackCapture> stack_capture{nullptr};
// The stack of the operator the executor is running on this thread.
thread_local uint32 running_op_stack = 0;

inline int64 now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// The stack an allocation is charged to: the producer of the Var being
// allocated, else the operator running, else the caller of the moment.
uint32 allocation_stack() {
    if (!(memory_history_mode & mh_alloc_stacks)) return 0;
    if (Var* v = step_trace_current_var())
        if (Op* op = v->input())
            if (op->memory_stack) return op->memory_stack;
    if (running_op_stack) return running_op_stack;
    return capture_memory_stack();
}

uint32 free_stack() {
    return (memory_history_mode & mh_free_stacks) ? capture_memory_stack() : 0;
}

} // namespace

void set_memory_stack_capture(MemoryStackCapture capture) { stack_capture.store(capture); }

uint32 capture_memory_stack() {
    auto capture = stack_capture.load();
    return capture ? capture() : 0;
}

uint32 memory_history_intern_frame(const char* file, int line, const char* name) {
    auto& s = state();
    string key = string(file) + '\n' + std::to_string(line) + '\n' + name;
    std::lock_guard<std::mutex> guard(s.mutex);
    auto it = s.frame_ids.find(key);
    if (it != s.frame_ids.end()) return it->second;
    uint32 id = s.frames.size() / 3;
    s.frames.push_back(file);
    s.frames.push_back(std::to_string(line));
    s.frames.push_back(name);
    s.frame_ids.emplace(move(key), id);
    return id;
}

uint32 memory_history_intern_stack(const uint32* frames, size_t count) {
    if (!count) return 0;
    auto& s = state();
    string key((const char*)frames, count * sizeof(uint32));
    std::lock_guard<std::mutex> guard(s.mutex);
    auto it = s.stack_ids.find(key);
    if (it != s.stack_ids.end()) return it->second;
    uint32 id = s.stacks.size();
    s.stacks.emplace_back(frames, frames + count);
    s.stack_ids.emplace(move(key), id);
    return id;
}

MemoryHistoryOpScope::MemoryHistoryOpScope(Op* op) : previous(running_op_stack) {
    running_op_stack = op ? op->memory_stack : 0;
}
MemoryHistoryOpScope::~MemoryHistoryOpScope() { running_op_stack = previous; }

// ---- Python API ----------------------------------------------------------

void memory_history_start(int64 mode, int64 max_entries) {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    // Stopping forgets the trace and the block state, as PyTorch's does; so
    // does `clear_history`, which keeps the block state.
    if (!mode) s.clear();
    if (mode & mh_clear_trace) s.traces.clear();
    s.max_entries = max_entries;
    memory_history_mode = (int)(mode & ~mh_clear_trace);
}

int64 memory_history_mode_now() { return memory_history_mode; }

string memory_history_layout() { return kLayout; }

vector<string> memory_history_frames() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    return s.frames;
}

vector<int64> memory_history_stacks() {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    vector<int64> out;
    for (size_t i = 1; i < s.stacks.size(); i++) {
        out.push_back(s.stacks[i].size());
        out.insert(out.end(), s.stacks[i].begin(), s.stacks[i].end());
    }
    return out;
}

vector<int64> memory_history_trace(int device) {
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    vector<int64> out;
    auto it = s.traces.find(device);
    if (it == s.traces.end()) return out;
    out.reserve(it->second.size() * kTraceFields);
    for (auto& e : it->second) {
        int64 rec[kTraceFields] = {e.action, e.addr, e.size, e.stream, e.time_us, e.stack, e.device_free};
        out.insert(out.end(), rec, rec + kTraceFields);
    }
    return out;
}

vector<int64> memory_history_devices() {
    std::set<int64> devices;
    for (SFRLAllocator* a : SFRLAllocator::sfrl_allocators)
        if (a->is_cuda() && a->device() >= 0) devices.insert(a->device());
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    for (auto& kv : s.traces) devices.insert(kv.first);
    return vector<int64>(devices.begin(), devices.end());
}

vector<int64> memory_history_segments(int device) {
    struct Block { int64 address, size, requested, state; uint32 stack; };
    struct Segment {
        int64 address = 0, total = 0, allocated = 0, requested = 0;
        uint32 stack = 0;
        bool small = false;
        vector<Block> blocks;
    };
    vector<Segment> segments;
    auto& s = state();
    for (SFRLAllocator* a : SFRLAllocator::sfrl_allocators) {
        if (!a->is_cuda() || a->device() != device) continue;
        // Read under the pool's own lock: alloc and free relink the chains.
        std::unique_lock<std::recursive_mutex> lock(a->mutex);
        vector<CachingBlock*> heads;
        auto add = [&](CachingBlock* b) { if (b && !b->prev) heads.push_back(b); };
        for (auto& kv : a->small_blocks.blocks) add(kv.second);
        for (auto& kv : a->large_blocks.blocks) add(kv.second);
        for (CachingBlock* b : a->id_space.occupied_id_mapper) add(b);
        std::lock_guard<std::mutex> guard(s.mutex);
        for (CachingBlock* head : heads) {
            Segment seg;
            seg.address = (int64)(uintptr_t)head->memory_ptr;
            auto known = s.segments.find((uint64)seg.address);
            if (known != s.segments.end()) seg.stack = known->second.stack;
            for (CachingBlock* b = head; b; b = b->next) {
                Block blk{(int64)(uintptr_t)b->memory_ptr, (int64)b->size, 0, b->occupied ? 1 : 0, 0};
                if (b->occupied) {
                    auto found = s.blocks.find((uint64)blk.address);
                    blk.requested = blk.size;
                    if (found != s.blocks.end()) {
                        blk.requested = found->second.requested;
                        blk.stack = found->second.stack;
                    }
                    seg.allocated += blk.size;
                    seg.requested += blk.requested;
                }
                seg.total += blk.size;
                seg.blocks.push_back(blk);
            }
            // A request of at most SMALL_BLOCK_SIZE gets a segment of exactly
            // that size (`allocation_size`); every other segment is larger.
            seg.small = seg.total <= (int64)SFRLAllocator::SMALL_BLOCK_SIZE;
            segments.push_back(move(seg));
        }
    }
    std::sort(segments.begin(), segments.end(),
              [](const Segment& x, const Segment& y) { return x.address < y.address; });
    vector<int64> out;
    for (auto& seg : segments) {
        int64 rec[kSegmentFields] = {seg.address, seg.total, seg.allocated, seg.requested, 0,
                                     seg.small, seg.stack, (int64)seg.blocks.size()};
        out.insert(out.end(), rec, rec + kSegmentFields);
        for (auto& b : seg.blocks) {
            int64 brec[kBlockFields] = {b.address, b.size, b.requested, b.state, b.stack};
            out.insert(out.end(), brec, brec + kBlockFields);
        }
    }
    std::lock_guard<std::mutex> guard(s.mutex);
    s.push(device, {mha_snapshot, 0, 0, 0, now_us(), 0, -1});
    return out;
}

// ---- allocator hooks -------------------------------------------------------

void memory_history_block(int action, int device, const void* ptr, int64 requested, int64 size) {
    if (device < 0) return;
    uint32 stack = action == mha_alloc ? allocation_stack() : free_stack();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    uint64 addr = (uint64)(uintptr_t)ptr;
    int64 t = now_us();
    if (action == mha_alloc) {
        s.blocks[addr] = {requested, stack};
        s.push(device, {mha_alloc, (int64)addr, requested, 0, t, stack, -1});
        return;
    }
    // A free: PyTorch records the request and its completion; the pools here
    // never defer a free behind a stream, so both happen now.
    auto found = s.blocks.find(addr);
    int64 nbytes = size;
    if (found != s.blocks.end()) {
        nbytes = found->second.requested;
        s.blocks.erase(found);
    }
    s.push(device, {mha_free_requested, (int64)addr, nbytes, 0, t, stack, -1});
    s.push(device, {mha_free_completed, (int64)addr, nbytes, 0, t, stack, -1});
}

void memory_history_segment(int action, int device, const void* ptr, int64 size) {
    if (device < 0) return;
    uint32 stack = action == mha_segment_alloc ? allocation_stack() : free_stack();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    uint64 addr = (uint64)(uintptr_t)ptr;
    if (action == mha_segment_alloc) s.segments[addr] = {size, stack};
    else s.segments.erase(addr);
    s.push(device, {action, (int64)addr, size, 0, now_us(), stack, -1});
}

void memory_history_oom(int device, int64 requested, int64 device_free) {
    if (device < 0) return;
    uint32 stack = allocation_stack();
    auto& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    s.push(device, {mha_oom, 0, requested, 0, now_us(), stack, device_free});
}

} // jittor
