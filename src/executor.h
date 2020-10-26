// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "mem/allocator.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "mem/allocator/mssfrl_allocator.h"
#endif

namespace jittor {
#ifdef HAS_CUDA
struct CudaEventPool {
    vector<cudaEvent_t*> events;
    int cnt;
    CudaEventPool() : cnt(0) {}

    ~CudaEventPool() {
        while (!events.empty()) {
            checkCudaErrors(cudaEventDestroy(*events.back()));
            delete events.back();
            events.pop_back();
        }
    }

    cudaEvent_t* get_event() {
        if (events.empty()) {
            ++cnt;
            cudaEvent_t* event = new cudaEvent_t();
            checkCudaErrors(cudaEventCreate(event));
            return event;
        }
        cudaEvent_t* event = events.back();
        events.pop_back();
        return event;
    }

    void recycle_event(cudaEvent_t* event) {
        events.push_back(event);
    }
};
#endif

struct Executor {
    #ifdef HAS_CUDA
    CudaEventPool cuda_event_pool;
    bool cuda_streams_initialized;
    static const int CUDA_STREAM_NUM = 5; //TODO chagne to 64. but now there are more than 64 streams, so set to 32.
    cudaStream_t cuda_streams[CUDA_STREAM_NUM];

    Executor() {
        cuda_streams_initialized = false;
    }

    void cuda_streams_init() {
        if (((MSSFRLAllocator*)allocator)->cuda_streams_initialized)
            return;
        cuda_streams[0] = 0;
        ((MSSFRLAllocator*)allocator)->set_stream_n(CUDA_STREAM_NUM);
        ((MSSFRLAllocator*)allocator)->streams_id_mapper[&cuda_streams[0]] = 0;
        for (int i = 1; i < CUDA_STREAM_NUM; ++i) {
            if (!cuda_streams_initialized) 
                checkCudaErrors(cudaStreamCreateWithFlags(&cuda_streams[i], cudaStreamNonBlocking));
            ((MSSFRLAllocator*)allocator)->streams_id_mapper[&cuda_streams[i]] = i;
        }
        ((MSSFRLAllocator*)allocator)->cuda_streams_initialized = true;
        cuda_streams_initialized = true;
    }
    #endif
    Allocator* allocator;
    bool last_is_cuda = false;
    void run_sync(vector<Var*> vars, bool device_sync);
    bool check_all_reduce(Op* op);
    bool check_log(Op* op);
};

extern Executor exe;

void load_fused_op(FusedOp& fused_op, vector<int>& fuse_ops, vector<Op*>& ops, int ll, int rr, int64 tt);
    
} // jittor