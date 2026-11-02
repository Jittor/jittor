// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

// The device memory history behind `torch.cuda.memory._record_memory_history`
// and `_snapshot` (python/jittor/profiling/_memory_history.py).
//
// While it is on, every block the device pools hand out or take back and every
// segment they get from or return to the driver is a trace entry, with the
// actions PyTorch's caching allocator records: segment_alloc, alloc,
// free_requested, free_completed, segment_free, oom, snapshot. A snapshot then
// reads the pools' current segments and blocks, so the pickle PyTorch's
// memory_viz reads can be written from Jittor's own allocator.
//
// Unlike `jt.profile`'s memory events this is not tied to a region: it stays
// on from start to stop, across any number of steps, keeping the newest
// `max_entries` entries per device.
//
// Stacks. Jittor executes lazily, so the Python stack at the moment a block is
// allocated is the line that forced execution, not the line that asked for the
// tensor. The stack recorded for a Var's block is therefore the one captured
// when its producing operator was built (`Op::memory_stack`); a workspace is
// charged to the operator running, and a free -- or an allocation outside the
// executor -- records the stack of the moment if the thread holds the GIL.

namespace jittor {

struct Op;

// 0 off; otherwise a MemoryHistoryMode bit set.
EXTERN_LIB int memory_history_mode;
// Nonzero while someone wants `Op::memory_stack` captured: bit 1 the memory
// history (allocation stacks), bit 2 a `jt.profile(stacks=True)` region.
EXTERN_LIB int capture_op_stacks;

enum MemoryHistoryMode {
    mh_on = 1,            // block state (and its stacks) for snapshots
    mh_trace = 2,         // the alloc/free/segment trace
    mh_alloc_stacks = 4,  // stacks for allocations
    mh_free_stacks = 8,   // stacks for frees too
    mh_clear_trace = 16,  // memory_history_start only: drop the trace so far
};

enum MemoryHistoryAction {
    mha_alloc = 0, mha_free_requested = 1, mha_free_completed = 2,
    mha_segment_alloc = 3, mha_segment_free = 4, mha_oom = 5, mha_snapshot = 6,
};

// A captured stack: an id into the stack table, 0 for none. Captures only when
// the calling thread holds the GIL; the binding layer installs the function.
typedef uint32 (*MemoryStackCapture)();
EXTERN_LIB void set_memory_stack_capture(MemoryStackCapture capture);
uint32 capture_memory_stack();
// For the capture function: intern a frame / a stack (innermost frame first).
// Ids stay valid for the life of the process.
uint32 memory_history_intern_frame(const char* file, int line, const char* name);
uint32 memory_history_intern_stack(const uint32* frames, size_t count);

// ---- Python API ----------------------------------------------------------

/// Start (or reconfigure) recording; `mode` is a MemoryHistoryMode bit set.
/// 0 stops and forgets the trace and the block state.
// @pyjt(memory_history_start)
void memory_history_start(int64 mode, int64 max_entries);
// @pyjt(memory_history_mode_now)
int64 memory_history_mode_now();
/// Turn op-stack capture on or off for one of its users (`capture_op_stacks` bits).
// @pyjt(set_op_stack_capture)
void set_op_stack_capture(int64 user, bool on);
/// "trace:field,...;segment:field,...;block:field,..."
// @pyjt(memory_history_layout)
string memory_history_layout();
/// Frames as (filename, line, name) triples, flattened; index = frame id.
// @pyjt(memory_history_frames)
vector<string> memory_history_frames();
/// Stacks: for each id from 1, its length then its frame ids.
// @pyjt(memory_history_stacks)
vector<int64> memory_history_stacks();
/// The recorded trace of `device`, flat; `memory_history_layout` names fields.
// @pyjt(memory_history_trace)
vector<int64> memory_history_trace(int device);
/// Devices that have device pools or trace entries, ascending.
// @pyjt(memory_history_devices)
vector<int64> memory_history_devices();
/// The device pools' segments and their blocks now, flat: each segment record
/// is followed by its `blocks` block records. Also appends a "snapshot" entry
/// to the trace, as PyTorch does.
// @pyjt(memory_history_segments)
vector<int64> memory_history_segments(int device);

// ---- allocator / executor hooks (C++ only) ---------------------------------

void memory_history_block(int action, int device, const void* ptr, int64 requested, int64 size);
void memory_history_segment(int action, int device, const void* ptr, int64 size);
void memory_history_oom(int device, int64 requested, int64 device_free);

// Around an operator's run in the executor: allocations made while it runs
// that are not for a Var (library workspaces) are charged to its stack.
struct MemoryHistoryOpScope {
    uint32 previous;
    explicit MemoryHistoryOpScope(Op* op);
    ~MemoryHistoryOpScope();
};

} // jittor
