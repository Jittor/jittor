// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include "common.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace jittor {

enum CudaSideStreamKind {
    CUDA_COPY_STREAM = 0,
    CUDA_COMMUNICATION_STREAM = 1,
};

// @pyjt(_cuda_stream_handle)
uint64 cuda_stream_handle(int kind, int device);

#ifdef HAS_CUDA
cudaStream_t cuda_side_stream(CudaSideStreamKind kind, int device);
void cuda_side_stream_wait_default(
    CudaSideStreamKind kind, int stream_device, int default_device);
void cuda_default_stream_wait_side(
    CudaSideStreamKind kind, int stream_device, int default_device);
#endif

} // namespace jittor
