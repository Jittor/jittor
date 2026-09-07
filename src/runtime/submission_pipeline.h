// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"

namespace jittor {

// Bookkeeping for *when* a graph gets submitted, as opposed to *how* it runs.
//
// Deciding to flush early is a scheduling policy layered on top of the
// executor, not part of executing a batch: the executor is handed a set of
// Vars and runs them, and nothing in that job depends on how many operators
// Python has built since last time. Keeping the two counters here rather than
// on `Executor` is what makes that true in the type system -- `executor.h`
// describes running a batch and nothing else, and a reader of `run_sync` no
// longer has to work out which members are inputs to execution and which are
// pipeline residue.
//
// Owned by `NativeRuntime`, alongside the executor it schedules for.
struct SubmissionPipeline {
    // Op::number_of_created_ops as of the most recent run_sync. The
    // auto-flush pipeline counts newly built operators from here, so its
    // flush points are anchored to executions and repeat identically across
    // steps -- drifting points would cut the graph differently every step
    // and compile a new fused-kernel variant each time.
    int64 last_run_ops = 0;
    // Python callbacks may return Vars while a submitted graph is executing;
    // submission must not nest through that conversion boundary.
    bool flush_active = false;
};

EXTERN_LIB SubmissionPipeline& runtime_submission_pipeline();

} // namespace jittor
